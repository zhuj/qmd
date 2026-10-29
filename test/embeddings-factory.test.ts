import { describe, expect, test } from "vitest";
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { LlamaCpp } from "../src/llm.js";
import type { EmbeddingResult } from "../src/llm.js";
import { createEmbeddingCapability } from "../src/embeddings/factory.js";
import type { LegacyEmbeddingFormatting } from "../src/embeddings/gguf.js";
import { resolveEmbeddingConfig } from "../src/embeddings/config.js";
import { makeOnnxConfig } from "./_helpers/embedding-config.js";
import { createArtifactIO } from "../src/embeddings/artifact-io.js";
import type { ResolvedEmbeddingConfig, EmbeddingRecipe, OnnxEmbeddingConfig, EmbeddingIdentity } from "../src/embeddings/types.js";
import type { ArtifactIO, ArtifactSnapshot } from "../src/embeddings/artifacts.js";
import type { OnnxRuntimePort, OnnxRuntimeSession, OnnxLoadOptions } from "../src/embeddings/onnx-runtime-types.js";
import type { EmbeddingTokenizer, EmbeddingModelOutput } from "../src/embeddings/recipe.js";
const formatting: LegacyEmbeddingFormatting = { query:(t,m)=>`q:${m}:${t}`, document:(t,title,m)=>`d:${m}:${title}:${t}`, fingerprint:(m)=>`fp:${m}` };
class FakeLlama extends LlamaCpp {
  texts: string[] = [];
  result: EmbeddingResult | null = { embedding: [0.25, 0.75], model: "fixture.gguf" };
  disposals = 0;
  constructor() { super({ embedModel: "fixture.gguf", inactivityTimeoutMs: 0 }); }
  override async embed(text: string): Promise<EmbeddingResult | null> { this.texts.push(text); return this.result; }
  override async embedBatch(texts: string[]): Promise<(EmbeddingResult | null)[]> { this.texts.push(...texts); return texts.map(() => this.result); }
  override async dispose(): Promise<void> { this.disposals++; }
}
function graph(): Buffer { return Buffer.from([8, 10, 58, 0]); }
function modelFiles(): readonly (readonly [string, Buffer])[] {
  return [
    ["config.json", Buffer.from(JSON.stringify({ model_type: "bert", hidden_size: 2, max_position_embeddings: 512 }))],
    ["tokenizer_config.json", Buffer.from('{"tokenizer_class":"BertTokenizer"}')],
    ["tokenizer.json", Buffer.from('{"model":{"type":"WordPiece","vocab":{"[UNK]":0},"unk_token":"[UNK]"}}')],
    ["onnx/model.onnx", graph()],
  ];
}
function tokenLength(text: string): number { return Math.max(1, text.length) + 2; }
interface FakeObservations {
  loads: Array<{ snapshot: ArtifactSnapshot; options: OnnxLoadOptions }>;
  runs: string[][];
  releases: number;
  closes: number;
  diagnostics: Array<{ code: string; requestedDevice: string; reason: string }>;
}
function fakePort(dimensions = 2): OnnxRuntimePort & { observations: FakeObservations } {
  const observations: FakeObservations = { loads: [], runs: [], releases: 0, closes: 0, diagnostics: [] };
  const tokenizer: EmbeddingTokenizer = {
    async encode(text, addSpecial) {
      const len = addSpecial ? tokenLength(text) : Math.max(1, text.length);
      return Array.from({ length: len }, (_, i) => i);
    },
    async decode(tokens) { return tokens.map(t => String.fromCharCode(65 + (t % 26))).join(""); },
  };
  const load = async (snapshot: ArtifactSnapshot, opts: OnnxLoadOptions): Promise<OnnxRuntimeSession> => {
    observations.loads.push({ snapshot, options: opts });
    return {
      tokenizer,
      async run(texts) {
        observations.runs.push([...texts]);
        const len = Math.max(...texts.map(t => Math.max(1, t.length) + 2));
        const mask = texts.map(t => Array.from({ length: len }, (_, i) => i < (Math.max(1, t.length) + 2) ? 1 : 0));
        const data = new Float32Array(texts.length * len * dimensions);
        for (let b = 0; b < texts.length; b++) {
          for (let s = 0; s < len; s++) {
            for (let d = 0; d < dimensions; d++) {
              data[b * len * dimensions + s * dimensions + d] = 0.1;
            }
          }
        }
        const output: EmbeddingModelOutput = {
          last_hidden_state: { data, dims: [texts.length, len, dimensions] },
        };
        return {
          output, mask,
          async release() { observations.releases++; },
        };
      },
      async close() { observations.closes++; },
    };
  };
  return {
    platform: process.platform, arch: process.arch,
    load, isProviderInitializationError: () => false,
    diagnostic: (e) => observations.diagnostics.push(e),
    observations,
  };
}
describe("T7d factory", () => {
  test("1) GGUF factory delegates createGguf and leases once; prepare/embed/close; borrowed llm untouched", async () => {
    const llm = new FakeLlama();
    const config: ResolvedEmbeddingConfig = { backend: "gguf", model: "fixture.gguf" };
    const cap = createEmbeddingCapability(config, llm, formatting);
    expect(cap.config).toEqual({ backend: "gguf", model: "fixture.gguf" });
    expect(cap.modelLabel).toBe("fixture.gguf");
    const identity = await cap.prepare();
    expect(identity.kind).toBe("legacy-gguf");
    const out = await cap.embed([{ kind: "query", text: "hello" }]);
    expect(out.length).toBe(1);
    expect(llm.texts).toContain("q:fixture.gguf:hello");
    await cap.close();
    expect(llm.disposals).toBe(0);
    await expect(cap.embed([{ kind: "query", text: "x" }])).rejects.toThrow();
  });
  test("2) ONNX factory uses injected IO/runtime, not borrowed LlamaCpp; prepare/embed/close; observations tracked", async () => {
    const root = await mkdtemp(resolve(".tmp", "onnx-factory-"));
    const local = join(root, "source");
    const work = join(root, "work");
    let cap: ReturnType<typeof createEmbeddingCapability> | undefined;
    try {
      for (const [p, b] of modelFiles()) {
        const dest = join(local, p);
        await mkdir(dirname(dest), { recursive: true });
        await writeFile(dest, b);
      }
      const recipe: EmbeddingRecipe = {
        version: "qmd-embedding-v1", pooling: "mean", normalization: "none",
        queryPrefix: "q: ", documentPrefix: "d: ", maxInputTokens: 128,
        titleBudgetTokens: 4, overlapTokens: 2,
      };
      const cfgValue = resolveEmbeddingConfig(
        makeOnnxConfig({ source: { kind: "local", path: local }, recipe }),
        { defaultGgufModel: "legacy", hostParallelism: 8 },
      );
      if (cfgValue.backend !== "onnx") throw new Error("expected onnx");
      const io = createArtifactIO(work);
      const port = fakePort(2);
      const llm = new FakeLlama();
      cap = createEmbeddingCapability(cfgValue, llm, formatting, io, port);
      expect(cap.config.backend).toBe("onnx");
      expect(cap.modelLabel).toBe("local ONNX model");
      await cap.prepare();
      const out = await cap.embed([{ kind: "query", text: "test" }]);
      expect(out.length).toBe(1);
      expect(llm.texts.length).toBe(0);
      expect(llm.disposals).toBe(0);
      expect(port.observations.loads.length).toBe(1);
      expect(port.observations.runs.length).toBe(2);
      expect(port.observations.releases).toBe(2);
      await cap.close();
      expect(port.observations.closes).toBe(1);
    } finally {
      try { if (cap !== undefined) await cap.close(); }
      finally { await rm(root, { recursive: true, force: true }); }
    }
  });
  test("3) lazy default construction: no prepare; default IO/runtime build; close safe", async () => {
    const root = await mkdtemp(resolve(".tmp", "onnx-factory-default-"));
    const local = join(root, "source");
    const work = join(root, "work");
    try {
      for (const [p, b] of modelFiles()) {
        const dest = join(local, p);
        await mkdir(dirname(dest), { recursive: true });
        await writeFile(dest, b);
      }
      const recipe: EmbeddingRecipe = {
        version: "qmd-embedding-v1", pooling: "mean", normalization: "none",
        queryPrefix: "q: ", documentPrefix: "d: ", maxInputTokens: 128,
        titleBudgetTokens: 4, overlapTokens: 2,
      };
      const cfgValue = resolveEmbeddingConfig(
        makeOnnxConfig({ source: { kind: "local", path: local }, recipe }),
        { defaultGgufModel: "legacy", hostParallelism: 8 },
      );
      if (cfgValue.backend !== "onnx") throw new Error("expected onnx");
      const llm = new FakeLlama();
      const cap = createEmbeddingCapability(cfgValue, llm, formatting);
      expect(cap.config.backend).toBe("onnx");
      expect(cap.modelLabel).toBe("local ONNX model");
      await cap.close();
      expect(llm.disposals).toBe(0);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
