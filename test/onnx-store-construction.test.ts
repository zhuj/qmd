import { afterEach, expect, test } from "vitest";
import { createStore as createSdkStore } from "../src/index.js";
import { configureStoreEmbeddings, createStore as createInternalStore } from "../src/store.js";
import type { EmbeddingCapability } from "../src/embeddings/types.js";
import { makeOnnxConfig } from "./_helpers/embedding-config.js";
import { createEmbeddingStoreFixture, fixtureEmbeddingConfig } from "./_helpers/embedding-store.js";
import { deferred } from "./_helpers/deferred.js";
import { loadConfig, setConfigSource } from "../src/collections.js";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
afterEach(() => setConfigSource(undefined));

test("lexical store construction and close never prepare or dispatch GGUF", async () => {
  const f = createEmbeddingStoreFixture();
  try {
    expect(f.store.searchFTS("alpha").map(result => result.filepath)).toEqual(["qmd://docs/alpha.md"]);
    expect(f.events).toEqual([]); expect(f.inputs).toEqual([]);
  } finally { await f.close(); }
  expect(f.events).toEqual(["close"]);
  await f.close(); expect(f.events).toEqual(["close"]);
});
test("per-store descriptor is copied, frozen and cannot be reconfigured", async () => {
  const parsed = fixtureEmbeddingConfig(), source: { kind: "local"; path: string } = { kind: "local", path: "/fixture/original" };
  const recipe = { ...parsed.recipe }, config = { ...parsed, source, recipe };
  const f = createEmbeddingStoreFixture(config);
  try {
    source.path = "/fixture/changed"; recipe.queryPrefix = "changed";
    expect(f.store.embeddingConfig).toMatchObject({ source: { path: "/fixture/original" }, recipe: { queryPrefix: "" } });
    expect(Object.isFrozen(f.store.embeddingConfig)).toBe(true);
    expect(Object.isFrozen(f.store.embeddingConfig?.backend === "onnx" && f.store.embeddingConfig.source)).toBe(true);
    expect(() => configureStoreEmbeddings(f.store, parsed)).toThrow(/immutable/i);
  } finally { await f.close(); }
});
test("closing one store preserves the other producer and lexical database", async () => {
  const a = createEmbeddingStoreFixture(fixtureEmbeddingConfig("/fixture/a"));
  const b = createEmbeddingStoreFixture(fixtureEmbeddingConfig("/fixture/b"));
  try {
    await a.close();
    expect(a.events).toEqual(["close"]); expect(b.events).toEqual([]);
    expect(b.store.embeddingConfig).toMatchObject({ source: { path: "/fixture/b" } });
    expect(b.store.searchFTS("alpha")).toHaveLength(1);
  } finally { await a.close(); await b.close(); }
});
test("SDK captures inline ONNX selection and lexical search stays lazy", async () => {
  const model = makeOnnxConfig({ source: { kind: "local", path: "/fixture/not-a-model" } });
  const store = await createSdkStore({ dbPath: ":memory:", config: { collections: {}, models: { embed: model } } });
  try {
    expect(store.internal.embeddingConfig).toMatchObject({ backend: "onnx", device: "cpu", source: { path: "/fixture/not-a-model" } });
    expect(await store.searchLex("alpha")).toEqual([]);
    expect(store.internal.embedding?.provenance).toBeUndefined();
  } finally { await store.close(); }
});
test("SDK rejects an unpinned model configuration before opening its database", async () => {
  await expect(createSdkStore({ dbPath: ":memory:", config: { collections: {}, models: {
    embed: makeOnnxConfig({ source: { kind: "huggingface", model: "fixture/model", revision: "main" } }),
  } } })).rejects.toThrow(/revision/);
});
test("sync close refuses an active lease; async close drains before capability and DB close", async () => {
  const config = fixtureEmbeddingConfig(), gate = deferred<readonly Float32Array[]>(), started = deferred<void>(), events: string[] = [];
  const cap: EmbeddingCapability = {
    config, modelLabel: "deferred", provenance: undefined,
    prepare: async () => ({ kind: "legacy-gguf", model: "unused", fingerprint: "unused", dimensions: 2 }),
    encode: async () => [], decode: async () => "",
    embed: async () => { started.resolve(); return gate.promise; },
    close: async () => { events.push("close"); },
  };
  const f = createEmbeddingStoreFixture(config, cap);
  const embedding = f.store.embedding;
  if (embedding === undefined) throw new Error("Missing configured producer");
  const pending = embedding.embed([{ kind: "query", text: "alpha" }]);
  try {
    await started.promise;
    expect(() => f.store.close()).toThrow(/closeAsync/);
    expect(f.store.searchFTS("alpha")).toHaveLength(1);
    const closing = f.close(); expect(events).toEqual([]);
    gate.resolve([new Float32Array([0.6, 0.8])]); await pending; await closing;
    expect(events).toEqual(["close"]);
    expect(() => f.store.searchFTS("alpha")).toThrow();
  } finally { gate.resolve([]); await pending; await f.close(); }
});
test("SDK closure does not reset a different live configuration source", async () => {
  const a = await createSdkStore({ dbPath: ":memory:", config: { collections: {} } });
  const expected = { collections: { retained: { path: "/fixture/retained", pattern: "**/*.md" } } };
  setConfigSource({ config: expected });
  try { await a.close(); expect(loadConfig()).toEqual(expected); }
  finally { await a.close(); setConfigSource(undefined); }
});
test("a configured producer requires async close even when idle", async () => {
  const f = createEmbeddingStoreFixture();
  try {
    expect(() => f.store.close()).toThrow(/closeAsync/);
    expect(f.store.searchFTS("alpha")).toHaveLength(1);
    expect(f.events).toEqual([]);
  } finally { await f.close(); }
});
test("store asynchronous close returns one stable completion promise", async () => {
  const f = createEmbeddingStoreFixture();
  const first = f.close(), second = f.close();
  try { expect(second).toBe(first); await first; expect(f.events).toEqual(["close"]); }
  finally { await f.close(); }
});
test("closing the exposed producer drains its admitted operation exactly once", async () => {
  const config = fixtureEmbeddingConfig(), gate = deferred<readonly Float32Array[]>(), started = deferred<void>();
  let closes = 0;
  const cap: EmbeddingCapability = {
    config, modelLabel: "deferred", provenance: undefined,
    prepare: async () => ({ kind: "legacy-gguf", model: "unused", fingerprint: "unused", dimensions: 2 }),
    encode: async () => [], decode: async () => "",
    embed: async () => { started.resolve(); return gate.promise; }, close: async () => { closes++; },
  };
  const f = createEmbeddingStoreFixture(config, cap), embedding = f.store.embedding;
  if (embedding === undefined) throw new Error("Missing producer");
  const pending = embedding.embed([{ kind: "query", text: "alpha" }]);
  try {
    await started.promise;
    const storeClosing = f.close();
    const closing = embedding.close(); expect(closes).toBe(0);
    expect(embedding.close()).toBe(closing);
    gate.resolve([new Float32Array([0.6, 0.8])]); await pending; await closing;
    await storeClosing; expect(closes).toBe(1);
  } finally { gate.resolve([]); await pending; await f.close(); }
});
test("pre-aborted capability calls reject asynchronously without starting producer work", async () => {
  const f = createEmbeddingStoreFixture(), controller = new AbortController(), reason = new Error("cancel fixture");
  controller.abort(reason);
  try {
    const embedding = f.store.embedding;
    if (embedding === undefined) throw new Error("Missing producer");
    await expect(embedding.prepare(controller.signal)).rejects.toBe(reason);
    expect(f.events).toEqual([]);
  } finally { await f.close(); }
});
test("independently ordered but equivalent capability config is admitted", async () => {
  const parsed = fixtureEmbeddingConfig(), { version, ...fields } = parsed.recipe;
  const reordered = { ...parsed, recipe: { ...fields, version } };
  const store = createInternalStore(":memory:");
  const capability: EmbeddingCapability = {
    config: reordered, modelLabel: "equivalent", provenance: undefined,
    prepare: async () => { throw new Error("Unexpected prepare"); },
    encode: async () => [], decode: async () => "", embed: async () => [], close: async () => {},
  };
  try {
    configureStoreEmbeddings(store, parsed, capability);
    expect(store.embeddingConfig).toEqual(parsed);
  } finally { await store.closeAsync(); }
});
test("file configuration resolves relative model paths without rewriting the YAML", async () => {
  const directory = await mkdtemp(resolve(".tmp", "onnx-store-config-")), configPath = join(directory, "index.yml");
  const input = JSON.stringify({ collections: {}, models: { embed: makeOnnxConfig({ source: { kind: "local", path: "./weights" } }) } });
  try {
    await writeFile(configPath, input);
    const store = await createSdkStore({ dbPath: ":memory:", configPath });
    try {
      expect(store.internal.embeddingConfig).toMatchObject({ source: { kind: "local", path: join(directory, "weights") } });
      expect(await store.searchLex("alpha")).toEqual([]);
      expect(await readFile(configPath, "utf8")).toBe(input);
    } finally { await store.close(); }
  } finally { await rm(directory, { recursive: true, force: true }); }
});
