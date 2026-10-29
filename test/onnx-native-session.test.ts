import { expect, test } from "vitest";
import { createNativeEmbeddingSession } from "../src/embeddings/onnx-native-session.js";
import { nativeFixture } from "./_helpers/onnx-native.js";
import { deferred } from "./_helpers/deferred.js";

test("runs the public native session with exact feeds and owns all tensors, without the logging wrapper", async () => {
  const f = nativeFixture(), session = createNativeEmbeddingSession(f.objects);
  const result = await session.run(["one", "two"]);
  expect(f.forwardCalls()).toBe(0); expect(f.feeds).toHaveLength(1);
  expect(f.feeds[0]).toMatchObject({ input_ids: f.input.input_ids.ort_tensor,
    attention_mask: f.input.attention_mask.ort_tensor, token_type_ids: { type: "int64", data: new BigInt64Array(4), dims: [2, 2] } });
  expect(f.calls).toEqual([[["one", "two"], { padding: true, truncation: false, return_tensor: true }]]);
  expect(result.mask).toEqual([[1, 1], [1, 0]]);
  expect(result.output.last_hidden_state?.data).toEqual(f.output.last_hidden_state.data);
  expect(Object.keys(result.output)).toEqual(["last_hidden_state"]);
  expect(f.disposed.size).toBe(0);
  const releasing = result.release(); expect(result.release()).toBe(releasing); await releasing;
  expect(f.disposed.size).toBe(5); expect([...f.disposed.values()]).toEqual([1, 1, 1, 1, 1]);
  await session.close(); expect(f.modelCloses()).toBe(1);
});
test("explicit special-token accounting and decode options preserve tokenizer semantics", async () => {
  const f = nativeFixture(), session = createNativeEmbeddingSession(f.objects);
  expect(await session.tokenizer.encode("abc", false)).toEqual([3]);
  expect(await session.tokenizer.encode("abc", true)).toEqual([101, 3, 102]);
  expect(await session.tokenizer.decode([1])).toBe("decoded");
  expect(f.tokenizerOptions).toEqual([{ add_special_tokens: false }, { add_special_tokens: true },
    { skip_special_tokens: false, clean_up_tokenization_spaces: false }]);
  await session.close();
});
test("inference failure is redacted and releases inputs/default tensors before model close", async () => {
  const f = nativeFixture(), secret = new Error("private token payload");
  f.raw.run = async () => { throw secret; };
  const session = createNativeEmbeddingSession(f.objects);
  const failure: unknown = await session.run(["one", "two"]).catch((error: unknown) => error);
  expect(failure).toEqual(new Error("ONNX inference failed")); expect(failure).not.toHaveProperty("cause");
  expect(f.disposed.size).toBe(3); await session.close(); expect(f.modelCloses()).toBe(1);
});
test("close drains an admitted run AND its output lease, then rejects new operations", async () => {
  const f = nativeFixture(), gate = deferred<typeof f.output>();
  f.raw.run = async () => gate.promise;
  const session = createNativeEmbeddingSession(f.objects), pending = session.run(["one", "two"]);
  const closing = session.close(); expect(session.close()).toBe(closing);
  expect(f.modelCloses()).toBe(0);
  gate.resolve(f.output); const result = await pending;
  expect(f.modelCloses()).toBe(0); await result.release(); await closing;
  expect(f.modelCloses()).toBe(1);
  await expect(session.run(["one"])).rejects.toThrow(/clos/i);
  await expect(session.tokenizer.encode("x", false)).rejects.toThrow(/clos/i);
  await expect(session.tokenizer.decode([1])).rejects.toThrow(/clos/i);
});
test.each([{ names: [] }, { names: ["unknown_input"] }, { names: ["attention_mask"] }])("refuses unsupported input layouts $names before tokenization", ({ names }) => {
  const f = nativeFixture(); f.raw.inputNames = names;
  expect(() => createNativeEmbeddingSession(f.objects)).toThrow(/contract|session|input/i);
  expect(f.calls).toHaveLength(0);
});
test("rejects multiple sessions without invoking a convenience model method", () => {
  const f = nativeFixture();
  expect(() => createNativeEmbeddingSession({ ...f.objects, model: { ...f.model, sessions: { model: f.raw, decoder: f.raw } } }))
    .toThrow(/contract|session/i);
  expect(f.forwardCalls()).toBe(0);
});
test.each([
  { change: (f: ReturnType<typeof nativeFixture>) => { f.input.attention_mask.ort_tensor.data = new BigInt64Array([1n, 2n, 1n, 0n]); } },
  { change: (f: ReturnType<typeof nativeFixture>) => { f.input.attention_mask.ort_tensor.data = new BigInt64Array(4); } },
  { change: (f: ReturnType<typeof nativeFixture>) => { f.input.input_ids.ort_tensor.data = new BigInt64Array([-1n, 2n, 3n, 0n]); } },
  { change: (f: ReturnType<typeof nativeFixture>) => { f.input.input_ids.ort_tensor.dims = [1, 4]; } },
])("invalid tokenizer tensor data fails before inference and releases admitted inputs", async ({ change }) => {
  const f = nativeFixture(); change(f);
  const session = createNativeEmbeddingSession(f.objects);
  await expect(session.run(["one", "two"])).rejects.toThrow("ONNX inference failed");
  expect(f.feeds).toHaveLength(0); expect(f.disposed.size).toBe(2); await session.close();
});
test.each([
  { change: (f: ReturnType<typeof nativeFixture>) => { f.output.last_hidden_state.type = "float16"; } },
  { change: (f: ReturnType<typeof nativeFixture>) => { f.output.last_hidden_state.dims = [2, 2, 3]; } },
  { change: (f: ReturnType<typeof nativeFixture>) => { f.output.last_hidden_state.location = "gpu-buffer"; } },
])("malformed embedding outputs release every used/unused tensor", async ({ change }) => {
  const f = nativeFixture(); change(f);
  const session = createNativeEmbeddingSession(f.objects);
  await expect(session.run(["one", "two"])).rejects.toThrow("ONNX inference failed");
  expect(f.disposed.size).toBe(5); expect([...f.disposed.values()]).toEqual([1, 1, 1, 1, 1]); await session.close();
});
test("logits are never used as embedding output", async () => {
  const f = nativeFixture();
  const model = { ...f.model, sessions: { model: { ...f.raw, run: async () => ({ logits: f.output.last_hidden_state }) } } };
  const session = createNativeEmbeddingSession({ ...f.objects, model });
  await expect(session.run(["one", "two"])).rejects.toThrow("ONNX inference failed");
  expect(f.disposed.size).toBe(4); await session.close();
});
test("an unused output alias cannot double-dispose a transformer-owned input", async () => {
  const f = nativeFixture();
  const model = { ...f.model, sessions: { model: { ...f.raw, run: async () => ({
    last_hidden_state: f.output.last_hidden_state, unused: f.input.input_ids.ort_tensor,
  }) } } };
  const session = createNativeEmbeddingSession({ ...f.objects, model }), result = await session.run(["one", "two"]);
  await result.release(); expect(f.disposed.size).toBe(4);
  expect([...f.disposed.values()]).toEqual([1, 1, 1, 1]); await session.close();
});
test("cleanup failure cannot replace an inference failure and all other owners still release", async () => {
  const f = nativeFixture(), originalDispose = f.input.input_ids.dispose;
  f.input.input_ids.dispose = () => { originalDispose(); throw new Error("private cleanup path"); };
  f.raw.run = async () => { throw new Error("private inference input"); };
  const session = createNativeEmbeddingSession(f.objects);
  await expect(session.run(["one", "two"])).rejects.toThrow("ONNX inference failed");
  expect(f.disposed.size).toBe(3); await session.close();
});
test("failed output release is redacted, idempotent and drains the model close", async () => {
  const f = nativeFixture(), originalDispose = f.output.last_hidden_state.dispose;
  f.output.last_hidden_state.dispose = () => { originalDispose(); throw new Error("private output path"); };
  const session = createNativeEmbeddingSession(f.objects), result = await session.run(["one", "two"]);
  const closing = session.close(), releasing = result.release();
  expect(result.release()).toBe(releasing);
  expect(await releasing.catch((error: unknown) => error)).toEqual(new Error("ONNX tensor cleanup failed"));
  await closing; expect(f.disposed.size).toBe(5); expect(f.modelCloses()).toBe(1);
});
