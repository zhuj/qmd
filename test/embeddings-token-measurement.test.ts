import { expect, test } from "vitest";
import { withFixture } from "./_helpers/onnx-capability.js";
import { deferred } from "./_helpers/deferred.js";
import { leaseEmbeddingCapability } from "../src/embeddings/lifecycle.js";
import { createEmbeddingStoreFixture } from "./_helpers/embedding-store.js";
import type { EmbeddingCapability, EmbeddingInput, EmbeddingTokenMeasurement } from "../src/embeddings/types.js";

function measure(capability: EmbeddingCapability, input: EmbeddingInput, signal?: AbortSignal): Promise<EmbeddingTokenMeasurement> {
  const operation = capability.measureTokens;
  if (!operation) return Promise.reject(new Error("Missing producer token measurement"));
  return operation.call(capability, input, signal);
}
test("producer counts its bounded-title formatting and raw text independently without extra inference", async () => withFixture(async f => {
  const runtime = f.port(), capability = f.capability(runtime);
  expect(await measure(capability, { kind: "query", text: "cat" })).toEqual({ text: 3, total: 8 });
  expect(await measure(capability, { kind: "document", text: "cat", title: "LONGTITLE" })).toEqual({ text: 3, total: 13 });
  expect(runtime.observations.runs).toHaveLength(1);
  expect(runtime.observations.tokenizations).toContainEqual({ text: "cat", special: false });
  expect(runtime.observations.tokenizations).toContainEqual({ text: "d: ABCD\ncat", special: true });
  await capability.embed([{ kind: "document", text: "cat", title: "LONGTITLE" }]);
  expect(runtime.observations.runs.at(-1)).toEqual(["d: ABCD\ncat"]);
  expect((await capability.encode("cat")).length).toBe(5);
}));
test("pre-aborted measurement never prepares and closed capabilities refuse it", async () => withFixture(async f => {
  const runtime = f.port(), capability = f.capability(runtime), controller = new AbortController();
  const reason = new Error("measurement cancelled"); controller.abort(reason);
  await expect(measure(capability, { kind: "query", text: "x" }, controller.signal)).rejects.toBe(reason);
  expect(runtime.observations.loads).toHaveLength(0);
  await capability.close();
  await expect(measure(capability, { kind: "query", text: "x" })).rejects.toThrow(/clos/i);
}));
test("lease forwarding drains admitted measurement and withholds cancelled results", async () => withFixture(async f => {
  const base = f.capability(f.port()), gate = deferred<EmbeddingTokenMeasurement>();
  let calls = 0;
  const owned = leaseEmbeddingCapability({ ...base, measureTokens: async () => { calls++; return gate.promise; } });
  const controller = new AbortController(), reason = new Error("lease measurement cancelled");
  const outcome = measure(owned, { kind: "query", text: "x" }, controller.signal).catch((error: unknown) => error);
  controller.abort(reason); const closing = owned.close();
  expect(calls).toBe(1); gate.resolve({ text: 1, total: 3 });
  expect(await outcome).toBe(reason); await closing;
  await expect(measure(owned, { kind: "query", text: "x" })).rejects.toThrow(/clos/i);
}));
test("raw producer withholds a measurement cancelled during uninterruptible tokenizer work", async () => withFixture(async f => {
  const runtime = f.port(), originalLoad = runtime.load, gate = deferred<void>(), started = deferred<void>();
  runtime.load = async (snapshot, options) => {
    const session = await originalLoad(snapshot, options), originalEncode = session.tokenizer.encode;
    return { ...session, tokenizer: { ...session.tokenizer, encode: async (text, special) => {
      if (text === "cancel-me" && !special) { started.resolve(); await gate.promise; }
      return originalEncode(text, special);
    } } };
  };
  const cap = f.capability(runtime), controller = new AbortController(), reason = new Error("raw count cancelled");
  const outcome = measure(cap, { kind: "query", text: "cancel-me" }, controller.signal).catch((error: unknown) => error);
  try { await started.promise; controller.abort(reason); gate.resolve(); expect(await outcome).toBe(reason); }
  finally { gate.resolve(); await outcome; }
}));
test("store-owned measurement is tracked and never calls a borrowed GGUF producer", async () => {
  const fixture = createEmbeddingStoreFixture();
  try {
    const producer = fixture.store.embedding;
    if (!producer) throw new Error("Fixture producer missing");
    expect(await measure(producer, { kind: "document", text: "cat", title: "title" })).toEqual({ text: 3, total: 11 });
    expect(fixture.events.some(event => event.startsWith("gguf-"))).toBe(false);
  } finally { await fixture.close(); }
});
