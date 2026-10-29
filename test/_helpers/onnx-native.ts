import type { TransformersSessionObjects } from "../../src/embeddings/onnx-native-session.js";
export function nativeFixture() {
  const disposed = new Map<object, number>();
  const tensor = (type: string, data: BigInt64Array | Float32Array, dims: readonly number[]) => {
    const value = { type, data, dims, location: "cpu", dispose() { disposed.set(value, (disposed.get(value) ?? 0) + 1); } };
    return value;
  };
  const wrapped = (data: BigInt64Array, dims: readonly number[]) => {
    const ort_tensor = tensor("int64", data, dims);
    return { ort_tensor, dispose: () => ort_tensor.dispose() };
  };
  let modelCloses = 0, forwardCalls = 0;
  const input = {
    input_ids: wrapped(new BigInt64Array([1n, 2n, 3n, 0n]), [2, 2]),
    attention_mask: wrapped(new BigInt64Array([1n, 1n, 1n, 0n]), [2, 2]),
  };
  const output = {
    last_hidden_state: tensor("float32", new Float32Array([1, 0, 0, 2, 3, 0, 0, 0]), [2, 2, 2]),
    pooler_output: tensor("float32", new Float32Array([1, 2, 3, 4]), [2, 2]),
  };
  const feeds: unknown[] = [], tokenizerOptions: unknown[] = [], calls: unknown[][] = [];
  const raw = { inputNames: ["input_ids", "attention_mask", "token_type_ids"], async run(feed: unknown) {
    feeds.push(feed); return output;
  } };
  const tokenizer = Object.assign((texts: unknown, options: unknown) => {
    calls.push([texts, options]); return input;
  }, { encode: (text: string, options: { add_special_tokens: boolean }) => {
    tokenizerOptions.push(options); return options.add_special_tokens ? [101, text.length, 102] : [text.length];
  }, decode: (_ids: readonly number[], options: unknown) => { tokenizerOptions.push(options); return "decoded"; } });
  const model = { sessions: { model: raw }, async dispose() { modelCloses++; },
    forward() { forwardCalls++; throw new Error("unsafe wrapper called"); } };
  const objects: TransformersSessionObjects = { tokenizer, model, int64Tensor: wrapped };
  return { objects, input, output, raw, model, feeds, calls, tokenizerOptions, disposed,
    tensor, wrapped, modelCloses: () => modelCloses, forwardCalls: () => forwardCalls };
}
