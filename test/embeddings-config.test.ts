import { expect, test, afterEach } from "vitest";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "fs";
import { resolve, join } from "path";
import YAML from "yaml";
import { resolveEmbeddingConfig, resolveSearchStagePolicy } from "../src/embeddings/config.js";
import { loadConfig, parseCollectionConfig, setConfigSource } from "../src/collections.js";

const context = { defaultGgufModel: "default.gguf", hostParallelism: 8 };
const config = {
  backend: "onnx", source: { kind: "huggingface", model: "fixture/model", revision: "a".repeat(40) },
  recipe: { version: "qmd-embedding-v1", pooling: "mean", normalization: "l2",
    queryPrefix: "", documentPrefix: "", maxInputTokens: 256, titleBudgetTokens: 32, overlapTokens: 24 },
};
afterEach(() => setConfigSource());

test("default ONNX is CPU-only and beats legacy environment", () => {
  const resolved = resolveEmbeddingConfig(config, { ...context, legacyEnv: "legacy.gguf" });
  expect(resolved).toMatchObject({ backend: "onnx", device: "cpu", fallback: "error", dtype: "fp32",
    allowDownload: false, intraOpNumThreads: 1, batchSize: 1, maxBatchTokens: 4096 });
  expect(resolveSearchStagePolicy(resolved, {})).toEqual({ expand: false, rerank: false });
  for (const request of [{ expand: true }, { rerank: true }]) {
    expect(() => resolveSearchStagePolicy(resolved, request)).toThrow(/CPU.*policy/i);
  }
});

test.each([
  { batchSize: 33 }, { batchSize: 0 }, { batchSize: 1.5 }, { batchSize: "1" },
  { maxBatchTokens: 65537 }, { maxBatchTokens: 0 }, { maxBatchTokens: 1.5 },
  { intraOpNumThreads: 9 }, { intraOpNumThreads: 0 }, { intraOpNumThreads: 1.5 },
  { dimensions: 0 }, { dimensions: 1.5 }, { dimensions: Infinity },
  { session_options: {} }, { surprise: true }, { backend: "other" },
  { device: "webgpu" }, { device: null }, { fallback: "auto" }, { dtype: "fp16" }, { allowDownload: "true" },
])("rejects invalid control %j", override => {
  expect(() => resolveEmbeddingConfig({ ...config, ...override }, context)).toThrow();
});

test.each([
  { kind: "huggingface", model: "https://example.com/model", revision: "a".repeat(40) },
  ...["main", "abc", "g".repeat(40)].map(revision => ({ ...config.source, revision })),
  ...["../model", "owner/../model", "owner//model", "owner/model.git", "owner/a--b", "owner/a..b", "-model"].map(model => ({ ...config.source, model })),
  { ...config.source, extra: true }, { kind: "local", path: "relative" },
  { kind: "local", path: "" }, { kind: "local", path: "/model", revision: "x" },
  { kind: "url", path: "/model" },
  { kind: "local", path: "https://example.com/model" },
])("rejects invalid source %j", source => {
  expect(() => resolveEmbeddingConfig({ ...config, source }, context)).toThrow();
});

test("file-backed local sources also reject URL selectors", () => {
  expect(() => resolveEmbeddingConfig({ ...config, source: { kind: "local", path: "https://example.com/model" } },
    { ...context, configPath: "/config/index.yml" })).toThrow(/local.*path/i);
});

test.each([
  { pooling: "logits" }, { version: "v2" }, { normalization: "auto" }, { queryPrefix: 1 },
  { documentPrefix: false }, { maxInputTokens: 0 }, { maxInputTokens: 1.5 },
  { titleBudgetTokens: -1 }, { titleBudgetTokens: 256 }, { titleBudgetTokens: 1.5 },
  { overlapTokens: -1 }, { overlapTokens: 224 }, { overlapTokens: 1.5 },
  { clsTokenIndex: 0 }, { pooling: "cls" }, { pooling: "cls", clsTokenIndex: -1 },
  { pooling: "cls", clsTokenIndex: 256 }, { pooling: "cls", clsTokenIndex: 0.5 },
  { sentenceOutput: "sentence_embedding" }, { pooling: "sentence-output" },
  { pooling: "sentence-output", sentenceOutput: "logits" },
  { reduction: { profile: "matryoshka" } }, { surprise: true },
])("rejects invalid recipe %j", override => {
  expect(() => resolveEmbeddingConfig({ ...config, recipe: { ...config.recipe, ...override } }, context)).toThrow();
});

test("accepts bounded controls and supported pooling recipes without claiming reduction", () => {
  for (const recipe of [config.recipe, { ...config.recipe, pooling: "last-token", titleBudgetTokens: 0 },
    { ...config.recipe, pooling: "cls", clsTokenIndex: 0 },
    { ...config.recipe, pooling: "sentence-output", sentenceOutput: "sentence_embedding" }]) {
    expect(resolveEmbeddingConfig({ ...config, recipe, batchSize: 32, maxBatchTokens: 65536,
      intraOpNumThreads: 8, dimensions: 384, dtype: "q8", allowDownload: true }, context))
      .toMatchObject({ dimensions: 384, batchSize: 32 });
  }
});

test("legacy strings and absence preserve selector precedence and independent stages", () => {
  expect(resolveEmbeddingConfig("own.gguf", { ...context, legacyEnv: "env.gguf" }))
    .toEqual({ backend: "gguf", model: "own.gguf" });
  expect(resolveEmbeddingConfig(undefined, { ...context, legacyEnv: "env.gguf" }))
    .toEqual({ backend: "gguf", model: "env.gguf" });
  expect(resolveEmbeddingConfig(undefined, context)).toEqual({ backend: "gguf", model: "default.gguf" });
  expect(resolveEmbeddingConfig("", { ...context, legacyEnv: "env.gguf" })).toEqual({ backend: "gguf", model: "env.gguf" });
  expect(resolveEmbeddingConfig(undefined, { ...context, legacyEnv: "" })).toEqual({ backend: "gguf", model: "default.gguf" });
  for (const device of ["cuda", "dml", "coreml"]) {
    const resolved = resolveEmbeddingConfig({ ...config, device, fallback: "cpu" }, context);
    expect(resolveSearchStagePolicy(resolved, {})).toEqual({ expand: true, rerank: true });
    expect(resolveSearchStagePolicy(resolved, { expand: false, rerank: true })).toEqual({ expand: false, rerank: true });
  }
  expect(resolveSearchStagePolicy(resolveEmbeddingConfig("legacy", context), { rerank: false }))
    .toEqual({ expand: true, rerank: false });
});

test.each(Object.keys(config.recipe))("requires explicit recipe field %s", field => {
  const recipe = Object.fromEntries(Object.entries(config.recipe).filter(([key]) => key !== field));
  expect(() => resolveEmbeddingConfig({ ...config, recipe }, context)).toThrow();
});

test.each([null, false, 123, [], {}, { ...config, source: null }, { ...config, recipe: null }]
  .map(value => ({ value })))("rejects malformed embedding descriptor %j", ({ value }) => {
  expect(() => resolveEmbeddingConfig(value, context)).toThrow();
});

test("resolved descriptor is deeply copied and frozen", () => {
  const input = structuredClone(config);
  const resolved = resolveEmbeddingConfig(input, context);
  input.recipe.queryPrefix = "changed";
  input.source.model = "other/model";
  expect(Object.isFrozen(resolved)).toBe(true);
  if (resolved.backend !== "onnx") throw new Error("expected ONNX");
  expect(resolved.recipe.queryPrefix).toBe("");
  expect(Object.isFrozen(resolved.recipe)).toBe(true);
  expect(Object.isFrozen(resolved.source)).toBe(true);
});

test("file-relative paths resolve purely; inline local paths must be absolute", () => {
  const value = { ...config, source: { kind: "local", path: "models/tiny" } };
  expect(resolveEmbeddingConfig(value, { ...context, configPath: "/config/index.yml" }))
    .toMatchObject({ source: { kind: "local", path: "/config/models/tiny" } });
  expect(resolveEmbeddingConfig({ ...value, source: { kind: "local", path: "/missing/model" } }, context))
    .toMatchObject({ source: { path: "/missing/model" } });
});

test("collection parser preserves established settings and accepts empty YAML", () => {
  const input = { global_context: "global", editor_uri: "editor:{path}", extension_setting: true,
    collections: { notes: { path: "/notes", pattern: "**/*.md", context: { "/": "notes" },
      ignore: ["drafts/**"], update: "echo update", includeByDefault: false } },
    models: { embed: config, rerank: "rerank.gguf", generate: "generate.gguf" } };
  expect(parseCollectionConfig(input)).toEqual(input);
  expect(parseCollectionConfig(null)).toEqual({ collections: {} });
});

test.each([{ path: "/wiki" }, { path: "/wiki", update: "git pull --ff-only" }])(
  "legacy collection without a pattern retains the Markdown default: %j", collection => {
    const input = { collections: { wiki: collection } };
    const before = structuredClone(input);
    expect(parseCollectionConfig(input).collections.wiki).toEqual({ ...collection, pattern: "**/*.md" });
    expect(input).toEqual(before);
    const dir = mkdtempSync(resolve(".tmp", "embeddings-legacy-config-"));
    const path = join(dir, "index.yml");
    try {
      const content = YAML.stringify(input);
      writeFileSync(path, content);
      setConfigSource({ configPath: path });
      expect(loadConfig().collections.wiki).toEqual({ ...collection, pattern: "**/*.md" });
      expect(readFileSync(path, "utf8")).toBe(content);
    } finally {
      setConfigSource();
      rmSync(dir, { recursive: true, force: true });
    }
  },
);

test.each([null, 123, false].map(pattern => ({ pattern })))(
  "an omitted pattern default does not accept malformed explicit patterns: %j", ({ pattern }) => {
    expect(() => parseCollectionConfig({ collections: { wiki: { path: "/wiki", pattern } } })).toThrow();
  },
);

test.each([false, [], "bad", { models: { embed: false } }, { models: { embed: { ...config, extra: true } } },
  { models: { generate: 1 } }, { models: { rerank: {} } }, { models: { surprise: true } },
  { collections: { bad: { path: 1, pattern: "*" } } }].map(value => ({ value })))("rejects malformed collection boundary %j", ({ value }) => {
  expect(() => parseCollectionConfig(value)).toThrow();
});

test("inline load copies without mutation, inferred defaults or saves", () => {
  const input = parseCollectionConfig({ collections: {}, models: { embed: config } });
  const before = structuredClone(input);
  setConfigSource({ config: input });
  const loaded = loadConfig();
  expect(loaded).toEqual(before);
  expect(loaded).not.toBe(input);
  loaded.global_context = "not saved";
  expect(loadConfig()).toEqual(before);
  expect(input).toEqual(before);
});

test("inline load validates plain-JS settings at the boundary", () => {
  const embedding = resolveEmbeddingConfig(config, context);
  if (embedding.backend !== "onnx") throw new Error("expected ONNX");
  setConfigSource({ config: { collections: {}, models: { embed: { ...embedding, batchSize: 33 } } } });
  expect(() => loadConfig()).toThrow(/batchSize/);
});

test("YAML load validates without rewriting or inferred defaults", () => {
  const dir = mkdtempSync(resolve(".tmp", "embeddings-config-"));
  const path = join(dir, "index.yml");
  try {
    const content = YAML.stringify({ collections: {}, models: { embed: { ...config, source: { kind: "local", path: "model" } } } });
    writeFileSync(path, content);
    setConfigSource({ configPath: path });
    expect(loadConfig().models?.embed).toMatchObject({ source: { path: join(dir, "model") } });
    expect(readFileSync(path, "utf8")).toBe(content);
    writeFileSync(path, YAML.stringify({ models: { embed: { ...config, device: "auto" } } }));
    expect(() => loadConfig()).toThrow();
  } finally {
    setConfigSource();
    rmSync(dir, { recursive: true, force: true });
  }
});
