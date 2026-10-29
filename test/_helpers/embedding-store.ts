import { createStore, configureStoreEmbeddings, upsertStoreCollection, type Store } from "../../src/store.js";
import { LlamaCpp } from "../../src/llm.js";
import { resolveEmbeddingConfig } from "../../src/embeddings/config.js";
import { createArtifactManifest, embeddingSpaceId } from "../../src/embeddings/identity.js";
import { formatEmbeddingInput, type EmbeddingTokenizer } from "../../src/embeddings/recipe.js";
import type { EmbeddingCapability, EmbeddingInput, EmbeddingSpaceDescriptor, ResolvedEmbeddingConfig, ResolvedOnnxConfig } from "../../src/embeddings/types.js";
import { makeOnnxConfig, makeSpaceDescriptor } from "./embedding-config.js";

export function fixtureEmbeddingConfig(path = "/fixture/model"): ResolvedOnnxConfig {
  const config = resolveEmbeddingConfig(makeOnnxConfig({ source: { kind: "local", path } }), {
    defaultGgufModel: "legacy", hostParallelism: 8,
  });
  if (config.backend !== "onnx") throw new Error("Wrong fixture backend");
  return config;
}
export class ForbiddenLlama extends LlamaCpp {
  constructor(private readonly events: string[]) { super({ inactivityTimeoutMs: 0 }); }
  override async tokenize(): Promise<never> { this.events.push("gguf-tokenize"); throw new Error("GGUF forbidden"); }
  override async detokenize(): Promise<never> { this.events.push("gguf-decode"); throw new Error("GGUF forbidden"); }
  override async embed(): Promise<never> { this.events.push("gguf-embed"); throw new Error("GGUF forbidden"); }
  override async embedBatch(): Promise<never> { this.events.push("gguf-batch"); throw new Error("GGUF forbidden"); }
  override async expandQuery(): Promise<never> { this.events.push("gguf-expand"); throw new Error("GGUF forbidden"); }
  override async rerank(): Promise<never> { this.events.push("gguf-rerank"); throw new Error("GGUF forbidden"); }
  override async dispose(): Promise<void> { this.events.push("gguf-dispose"); }
}
export interface EmbeddingStoreFixture {
  readonly store: Store;
  readonly inputs: EmbeddingInput[];
  readonly events: string[];
  close(): Promise<void>;
}
export function createEmbeddingStoreFixture(
  config: ResolvedEmbeddingConfig = fixtureEmbeddingConfig(), capability?: EmbeddingCapability,
): EmbeddingStoreFixture {
  const store = createStore(":memory:"), events: string[] = [], inputs: EmbeddingInput[] = [];
  store.llm = new ForbiddenLlama(events);
  const base = makeSpaceDescriptor();
  const descriptor: EmbeddingSpaceDescriptor = config.backend === "onnx" ? {
    ...base, artifactManifest: createArtifactManifest(config.source, base.artifactManifest.files), recipe: config.recipe,
  } : base;
  const tokenizer: EmbeddingTokenizer = {
    encode: async (text, special) => special ? [0, ...Array.from(text, character => character.codePointAt(0) ?? 0), 0]
      : Array.from(text, character => character.codePointAt(0) ?? 0),
    decode: async tokens => String.fromCodePoint(...tokens),
  };
  const raw: EmbeddingCapability = capability ?? {
    config, modelLabel: "fixture-embedding", provenance: undefined,
    prepare: async () => {
      events.push("prepare");
      if (config.backend === "gguf") return { kind: "legacy-gguf", model: config.model, fingerprint: "fixture", dimensions: 2 };
      return { kind: "onnx-v1", id: embeddingSpaceId(descriptor), descriptor };
    },
    encode: text => tokenizer.encode(text, true),
    decode: tokens => tokenizer.decode(tokens),
    ...(config.backend === "onnx" ? { measureTokens: async (input: EmbeddingInput, signal?: AbortSignal) => {
      signal?.throwIfAborted();
      const formatted = await formatEmbeddingInput(input, config.recipe, tokenizer);
      return { text: (await tokenizer.encode(input.text, false)).length, total: (await tokenizer.encode(formatted, true)).length };
    } } : {}),
    embed: async values => {
      events.push("embed"); inputs.push(...values.map(value => ({ ...value })));
      return values.map(() => new Float32Array([0.6, 0.8]));
    },
    close: async () => { events.push("close"); },
  };
  try {
    upsertStoreCollection(store.db, "docs", { path: "/fixture/docs", pattern: "**/*.md" });
    const date = "2026-01-01T00:00:00Z", hash = "fixture-alpha-beta";
    store.insertContent(hash, "alpha beta", date);
    store.insertDocument("docs", "alpha.md", "Alpha", hash, date, date);
    configureStoreEmbeddings(store, config, raw);
  } catch (error) { store.close(); throw error; }
  return { store, events, inputs, close: () => store.closeAsync() };
}
