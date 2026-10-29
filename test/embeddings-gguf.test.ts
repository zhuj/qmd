import { describe, expect, test } from "vitest";
import type { Token } from "node-llama-cpp";
import { canUnloadLLM, LlamaCpp, withLLMSession, withLLMSessionForLlm, type EmbeddingResult } from "../src/llm.js";
import { createGgufEmbeddingCapability, type LegacyEmbeddingFormatting } from "../src/embeddings/gguf.js";
import { leaseEmbeddingCapability } from "../src/embeddings/lifecycle.js";
import { deferred } from "./_helpers/deferred.js";

class FixtureLlama extends LlamaCpp {
  readonly texts: string[] = [];
  result: EmbeddingResult | null = { embedding: [0.25, 0.75], model: "fixture.gguf" };
  disposals = 0;
  constructor() { super({ embedModel: "fixture.gguf", inactivityTimeoutMs: 0 }); }
  override async tokenize(text: string): Promise<readonly Token[]> {
    // The fake preserves the same numeric representation as the native opaque token type.
    return Array.from(text, (character) => character.codePointAt(0) ?? 0) as Token[];
  }
  override async detokenize(tokens: readonly Token[]): Promise<string> { return String.fromCodePoint(...tokens); }
  override async embed(text: string): Promise<EmbeddingResult | null> { this.texts.push(text); return this.result; }
  override async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> {
    this.texts.push(...texts);
    return texts.map(() => this.result);
  }
  override async dispose(): Promise<void> { this.disposals++; }
}

const formatting: LegacyEmbeddingFormatting = {
  query: (text, model) => `query:${model}:${text}`,
  document: (text, title, model) => `doc:${model}:${title}:${text}`,
  fingerprint: (model) => `legacy:${model}`,
};

describe("GGUF embedding capability", () => {
  test("legacy no-argument guard still tracks nested default sessions without native inference", async () => {
    await withLLMSession(async () => {
      expect(canUnloadLLM()).toBe(false);
      await withLLMSession(async () => { expect(canUnloadLLM()).toBe(false); });
      expect(canUnloadLLM()).toBe(false);
    });
    expect(canUnloadLLM()).toBe(true);
  });

  test("instance sessions prevent idle unloading without holding another store's model", async () => {
    const llm = new FixtureLlama();
    const other = new FixtureLlama();
    await withLLMSessionForLlm(llm, async () => {
      expect(canUnloadLLM(llm)).toBe(false);
      expect(canUnloadLLM(other)).toBe(true);
      await withLLMSessionForLlm(llm, async () => { expect(canUnloadLLM(llm)).toBe(false); });
      expect(canUnloadLLM(llm)).toBe(false);
    });
    expect(canUnloadLLM(llm)).toBe(true);
  });

  test("preflight remains leased through inflight abort and releases after native work settles", async () => {
    const run = deferred<EmbeddingResult | null>();
    class DeferredLlama extends FixtureLlama {
      override async embed(): Promise<EmbeddingResult | null> { return run.promise; }
    }
    const llm = new DeferredLlama();
    const capability = leaseEmbeddingCapability(createGgufEmbeddingCapability(llm, "fixture.gguf", formatting));
    const controller = new AbortController();
    const preparing = capability.prepare(controller.signal);
    expect(canUnloadLLM(llm)).toBe(false);
    controller.abort(new Error("stop preflight"));
    const outcome = preparing.catch((error: unknown) => error);
    const closing = capability.close();
    expect(canUnloadLLM(llm)).toBe(false);
    run.resolve({ embedding: [1, 0], model: "fixture.gguf" });
    expect(await outcome).toEqual(new Error("stop preflight"));
    await closing;
    expect(canUnloadLLM(llm)).toBe(true);
    expect(llm.disposals).toBe(0);
  });

  test("prepares the actual producer's dimension and unchanged injected legacy fingerprint", async () => {
    const llm = new FixtureLlama();
    const capability = createGgufEmbeddingCapability(llm, "fixture.gguf", formatting);
    expect(await capability.prepare()).toEqual({
      kind: "legacy-gguf", model: "fixture.gguf", fingerprint: "legacy:fixture.gguf", dimensions: 2,
    });
    await capability.prepare();
    expect(llm.texts).toHaveLength(1);
  });

  test("rejects a selected model that differs from the actual loader before any inference", () => {
    const llm = new FixtureLlama();
    expect(() => createGgufEmbeddingCapability(llm, "other.gguf", formatting)).toThrow(/model/i);
    expect(llm.texts).toEqual([]);
  });

  test("formats raw query/document inputs once and preserves vector values and order", async () => {
    const llm = new FixtureLlama();
    const capability = createGgufEmbeddingCapability(llm, "fixture.gguf", formatting);
    await capability.prepare();
    llm.texts.length = 0;
    expect(await capability.embed([
      { kind: "query", text: "question" }, { kind: "document", text: "body", title: "heading" },
    ])).toEqual([new Float32Array([0.25, 0.75]), new Float32Array([0.25, 0.75])]);
    expect(llm.texts).toEqual(["query:fixture.gguf:question", "doc:fixture.gguf:heading:body"]);
  });

  test("uses the selected tokenizer and accepts its numeric token slices", async () => {
    const capability = createGgufEmbeddingCapability(new FixtureLlama(), "fixture.gguf", formatting);
    expect(await capability.encode("ab")).toEqual([97, 98]);
    expect(await capability.decode([98])).toBe("b");
  });

  test.each([-1, 1.5, NaN, Infinity, 2 ** 31])("rejects invalid native token id %s", async (token) => {
    const capability = createGgufEmbeddingCapability(new FixtureLlama(), "fixture.gguf", formatting);
    await expect(capability.decode([token])).rejects.toThrow(/token/i);
  });

  test.each([
    { name: "missing", result: null },
    { name: "wrong producer", result: { model: "other.gguf", embedding: [1, 0] } },
    { name: "empty", result: { model: "fixture.gguf", embedding: [] } },
    { name: "zero", result: { model: "fixture.gguf", embedding: [0, 0] } },
    { name: "nonfinite", result: { model: "fixture.gguf", embedding: [NaN, 1] } },
    { name: "float32 overflow", result: { model: "fixture.gguf", embedding: [1e99, 1] } },
  ])("rejects $name preflight vectors instead of inventing a dimension", async ({ result }) => {
    const llm = new FixtureLlama();
    llm.result = result;
    const capability = createGgufEmbeddingCapability(llm, "fixture.gguf", formatting);
    await expect(capability.prepare()).rejects.toThrow(/embedding|producer/i);
  });

  test("rejects changed dimensions after preparation", async () => {
    const llm = new FixtureLlama();
    const capability = createGgufEmbeddingCapability(llm, "fixture.gguf", formatting);
    await capability.prepare();
    llm.result = { model: "fixture.gguf", embedding: [1, 0, 0] };
    await expect(capability.embed([{ kind: "query", text: "x" }])).rejects.toThrow(/dimension/i);
  });

  test("a failed preflight can be retried without retaining an idle lease", async () => {
    const llm = new FixtureLlama();
    llm.result = null;
    const capability = createGgufEmbeddingCapability(llm, "fixture.gguf", formatting);
    await expect(capability.prepare()).rejects.toThrow(/embedding/i);
    expect(canUnloadLLM(llm)).toBe(true);
    llm.result = { embedding: [1, 0], model: "fixture.gguf" };
    expect(await capability.prepare()).toMatchObject({ dimensions: 2 });
  });

  test("one cancelled preflight caller does not poison another caller's shared initialization", async () => {
    const run = deferred<EmbeddingResult | null>();
    let probes = 0;
    class DeferredLlama extends FixtureLlama {
      override async embed(): Promise<EmbeddingResult | null> { probes++; return run.promise; }
    }
    const llm = new DeferredLlama();
    const capability = leaseEmbeddingCapability(createGgufEmbeddingCapability(llm, "fixture.gguf", formatting));
    const controller = new AbortController();
    const first = capability.prepare(controller.signal);
    const second = capability.prepare();
    controller.abort(new Error("caller cancelled"));
    const outcome = first.catch((error: unknown) => error);
    run.resolve({ embedding: [1, 0], model: "fixture.gguf" });
    expect(await outcome).toEqual(new Error("caller cancelled"));
    expect(await second).toMatchObject({ dimensions: 2 });
    expect(probes).toBe(1);
    await capability.close();
  });

  test("missing batch rows fail instead of silently dropping inputs", async () => {
    class ShortBatchLlama extends FixtureLlama {
      override async embedBatch(): Promise<(EmbeddingResult | null)[]> { return []; }
    }
    const capability = createGgufEmbeddingCapability(new ShortBatchLlama(), "fixture.gguf", formatting);
    await expect(capability.embed([{ kind: "query", text: "x" }])).rejects.toThrow(/count/i);
  });

  test("borrowed llama ownership remains with its store after capability close", async () => {
    const llm = new FixtureLlama();
    const capability = leaseEmbeddingCapability(createGgufEmbeddingCapability(llm, "fixture.gguf", formatting));
    await capability.close();
    expect(llm.disposals).toBe(0);
  });
});
