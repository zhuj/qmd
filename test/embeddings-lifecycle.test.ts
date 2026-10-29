import { describe, expect, test } from "vitest";
import { leaseEmbeddingCapability } from "../src/embeddings/lifecycle.js";
import type { EmbeddingCapability } from "../src/embeddings/types.js";
import { deferred } from "./_helpers/deferred.js";

function rawCapability(overrides: Partial<EmbeddingCapability> = {}): EmbeddingCapability {
  return {
    config: { backend: "gguf", model: "fixture.gguf" }, modelLabel: "fixture.gguf",
    provenance: undefined,
    prepare: async () => ({ kind: "legacy-gguf", model: "fixture.gguf", fingerprint: "abc123", dimensions: 2 }),
    encode: async () => [7], decode: async () => "text", embed: async () => [new Float32Array([1, 0])],
    close: async () => {}, ...overrides,
  };
}

describe("embedding capability leases", () => {
  test("close waits for admitted inference and disposes once", async () => {
    const run = deferred<readonly Float32Array[]>();
    let closes = 0;
    const capability = leaseEmbeddingCapability(rawCapability({
      embed: () => run.promise, close: async () => { closes++; },
    }));
    const inference = capability.embed([{ kind: "query", text: "first" }]);
    const closing = capability.close();
    expect(closes).toBe(0);
    await expect(capability.embed([{ kind: "query", text: "late" }])).rejects.toThrow(/clos/i);
    run.resolve([new Float32Array([1, 0])]);
    expect(await inference).toEqual([new Float32Array([1, 0])]);
    await closing;
    await capability.close();
    expect(closes).toBe(1);
  });

  test("close waits for tokenizer work too", async () => {
    const run = deferred<readonly number[]>();
    let closes = 0;
    const capability = leaseEmbeddingCapability(rawCapability({
      encode: () => run.promise, close: async () => { closes++; },
    }));
    const encoding = capability.encode("text");
    const closing = capability.close();
    expect(closes).toBe(0);
    run.resolve([4, 9]);
    expect(await encoding).toEqual([4, 9]);
    await closing;
    expect(closes).toBe(1);
  });

  test("already aborted work never enters the raw capability", async () => {
    const controller = new AbortController();
    controller.abort(new Error("fixture abort"));
    let calls = 0;
    const capability = leaseEmbeddingCapability(rawCapability({ embed: async () => { calls++; return []; } }));
    await expect(capability.embed([], controller.signal)).rejects.toThrow("fixture abort");
    expect(calls).toBe(0);
    await capability.close();
  });

  test("inflight cancellation withholds results but waits before disposal", async () => {
    const run = deferred<readonly Float32Array[]>();
    const controller = new AbortController();
    let closes = 0;
    const capability = leaseEmbeddingCapability(rawCapability({
      embed: () => run.promise, close: async () => { closes++; },
    }));
    const inference = capability.embed([], controller.signal);
    controller.abort(new Error("cancelled result"));
    const outcome = inference.catch((error: unknown) => error);
    const closing = capability.close();
    expect(closes).toBe(0);
    run.resolve([new Float32Array([1, 0])]);
    expect(await outcome).toEqual(new Error("cancelled result"));
    await closing;
    expect(closes).toBe(1);
  });

  test("raw failures release their lease so close can settle", async () => {
    const run = deferred<readonly Float32Array[]>();
    let closes = 0;
    const capability = leaseEmbeddingCapability(rawCapability({
      embed: () => run.promise, close: async () => { closes++; },
    }));
    const inference = capability.embed([]);
    const outcome = inference.catch((error: unknown) => error);
    const closing = capability.close();
    run.reject(new Error("native fixture failure"));
    expect(await outcome).toEqual(new Error("native fixture failure"));
    await closing;
    expect(closes).toBe(1);
  });

  test("concurrent close callers observe the same disposal failure", async () => {
    let closes = 0;
    const capability = leaseEmbeddingCapability(rawCapability({ close: async () => {
      closes++;
      throw new Error("release failed");
    } }));
    const first = capability.close();
    const second = capability.close();
    expect(first).toBe(second);
    await expect(first).rejects.toThrow("release failed");
    await expect(second).rejects.toThrow("release failed");
    expect(closes).toBe(1);
  });

  test.each(["prepare", "encode", "decode", "embed"] as const)("closed capability rejects %s", async (method) => {
    const capability = leaseEmbeddingCapability(rawCapability());
    await capability.close();
    const calls = {
      prepare: () => capability.prepare(), encode: () => capability.encode("x"),
      decode: () => capability.decode([7]), embed: () => capability.embed([]),
    };
    await expect(calls[method]()).rejects.toThrow(/clos/i);
  });

  test("closing one wrapper cannot close another store's capability", async () => {
    let secondCloses = 0;
    const first = leaseEmbeddingCapability(rawCapability());
    const second = leaseEmbeddingCapability(rawCapability({ close: async () => { secondCloses++; } }));
    await first.close();
    expect(await second.encode("x")).toEqual([7]);
    expect(secondCloses).toBe(0);
    await second.close();
    expect(secondCloses).toBe(1);
  });
});
