import { afterEach, expect, test } from "vitest";
import { createStore } from "../src/index.js";
import { makeOnnxConfig, makeSpaceDescriptor } from "./_helpers/embedding-config.js";
import { fixtureEmbeddingConfig, ForbiddenLlama } from "./_helpers/embedding-store.js";
import { configureStoreEmbeddings, upsertStoreCollection } from "../src/store.js";
import { createArtifactManifest, embeddingSpaceId } from "../src/embeddings/identity.js";
import { measureEmbeddingInput } from "../src/embeddings/recipe.js";
import { setConfigSource } from "../src/collections.js";
import { deferred } from "./_helpers/deferred.js";
import type { EmbeddingCapability, EmbeddingInput, ResolvedOnnxConfig } from "../src/embeddings/types.js";

afterEach(() => setConfigSource(undefined));

async function sdkFixture(path: string, beforeEmbed?: (inputs: readonly EmbeddingInput[]) => Promise<void>, selectedConfig?: ResolvedOnnxConfig) {
  const sdk = await createStore({ dbPath: ":memory:" });
  const config = selectedConfig ?? fixtureEmbeddingConfig(path), inputs: EmbeddingInput[] = [], events: string[] = [];
  const base = makeSpaceDescriptor();
  const descriptor = { ...base, artifactManifest: createArtifactManifest(config.source, base.artifactManifest.files), recipe: config.recipe };
  const tokenizer = { encode: async (text: string, special: boolean) => [
    ...(special ? [0] : []), ...Array.from(text, value => value.codePointAt(0) ?? 0), ...(special ? [0] : []),
  ], decode: async (tokens: readonly number[]) => String.fromCodePoint(...tokens) };
  const capability: EmbeddingCapability = {
    config, modelLabel: `sdk:${path}`, provenance: undefined,
    prepare: async () => ({ kind: "onnx-v1", id: embeddingSpaceId(descriptor), descriptor }),
    encode: text => tokenizer.encode(text, true), decode: tokenizer.decode,
    measureTokens: input => measureEmbeddingInput(input, config.recipe, tokenizer),
    embed: async values => { inputs.push(...values); await beforeEmbed?.(values); return values.map(() => new Float32Array([0.6, 0.8])); },
    close: async () => { events.push("embedding-close"); },
  };
  try {
    sdk.internal.llm = new ForbiddenLlama(events);
    upsertStoreCollection(sdk.internal.db, "docs", { path: "/fixture/docs" });
    sdk.internal.insertContent("sdk-fixture-hash", "alpha beta", "2026-01-01");
    sdk.internal.insertDocument("docs", "alpha.md", "Alpha", "sdk-fixture-hash", "2026-01-01", "2026-01-01");
    configureStoreEmbeddings(sdk.internal, config, capability);
    return { sdk, inputs, events, config };
  } catch (error) { await sdk.close(); throw error; }
}

test("public default CPU search preserves omitted stages, while explicit enable and manual expansion reject", async () => {
  const store = await createStore({ dbPath: ":memory:", config: { collections: {}, models: { embed: makeOnnxConfig() } } });
  try {
    expect(await store.search({ query: "alpha" })).toEqual([]);
    expect(await store.search({ queries: [{ type: "lex", query: "alpha" }] })).toEqual([]);
    expect(await store.search({ query: "alpha", rerank: false })).toEqual([]);
    await expect(store.search({ query: "alpha", rerank: true })).rejects.toThrow(/CPU.*policy/i);
    await expect(store.expandQuery("alpha")).rejects.toThrow(/CPU.*policy/i);
    expect(await store.searchLex("alpha")).toEqual([]);
  } finally { await store.close(); }
});
test("SDK JavaScript callers cannot turn a non-boolean rerank value into an implicit stage request", async () => {
  const store = await createStore({ dbPath: ":memory:", config: { collections: {}, models: { embed: makeOnnxConfig() } } });
  try {
    await expect(store.search(JSON.parse('{"query":"alpha","rerank":"false"}'))).rejects.toThrow(/rerank.*boolean/i);
  } finally { await store.close(); }
});
test("malformed plain-JavaScript ONNX config rejects at SDK admission", async () => {
  await expect(createStore({ dbPath: ":memory:", config: JSON.parse('{"collections":{},"models":{"embed":{"backend":"onnx","source":{"kind":"huggingface","model":"fixture/model","revision":"main"}}}}') }))
    .rejects.toThrow(/revision|recipe|Invalid/i);
});
test("SDK index/direct/hybrid/structured share one fake producer through the advanced internal seam", async () => {
  const f = await sdkFixture("/fixture/sdk-model");
  try {
    expect((await f.sdk.embed()).chunksEmbedded).toBe(1);
    f.inputs.length = 0;
    expect((await f.sdk.searchVector("alpha"))[0]?.filepath).toBe("qmd://docs/alpha.md");
    expect((await f.sdk.search({ query: "alpha" }))[0]?.file).toBe("qmd://docs/alpha.md");
    expect((await f.sdk.search({ queries: [{ type: "vec", query: "beta" }] }))[0]?.file).toBe("qmd://docs/alpha.md");
    expect(f.inputs).toEqual([{ kind: "query", text: "alpha" }, { kind: "query", text: "alpha" }, { kind: "query", text: "beta" }]);
    expect(f.events).toEqual([]);
    const calls = f.inputs.length;
    await expect(f.sdk.embed({ force: true, model: "other.gguf" })).rejects.toThrow(/model|producer/i);
    expect(f.inputs.length).toBe(calls);
  } finally { await f.sdk.close(); }
  expect(f.events).toEqual(["embedding-close", "gguf-dispose"]);
});
test("SDK embed preserves caller cancellation before any metadata or vector write", async () => {
  const started = deferred<void>(), release = deferred<void>(), controller = new AbortController();
  const f = await sdkFixture("/fixture/sdk-cancel", async () => { started.resolve(); await release.promise; });
  let pending: Promise<unknown> | undefined;
  try {
    pending = f.sdk.embed({ signal: controller.signal }).catch((error: unknown) => error);
    const phase = await Promise.race([started.promise.then(() => "started"), pending.then(() => "settled")]);
    expect(phase).toBe("started"); controller.abort("SDK cancelled"); release.resolve();
    expect(await pending).toBe("SDK cancelled");
    expect(f.sdk.internal.db.prepare("SELECT * FROM content_vectors").all()).toEqual([]);
    expect(f.sdk.internal.db.prepare("SELECT * FROM store_config WHERE key IN ('embedding_space_v1', 'embedding_runtime_v1')").all()).toEqual([]);
  } finally { release.resolve(); await pending; await f.sdk.close(); }
});
test("overlapping SDK lifetimes keep producer/close ownership despite a foreign global config source", async () => {
  const bBase = fixtureEmbeddingConfig("/fixture/sdk-b");
  const bConfig: ResolvedOnnxConfig = { ...bBase, device: "cuda", recipe: { ...bBase.recipe, queryPrefix: "second: " } };
  const a = await sdkFixture("/fixture/sdk-a");
  let b: Awaited<ReturnType<typeof sdkFixture>> | undefined;
  try {
    b = await sdkFixture("/fixture/sdk-b", undefined, bConfig);
    await a.sdk.embed(); await b.sdk.embed();
    setConfigSource({ config: { collections: {}, models: { embed: "foreign.gguf" } } });
    await a.sdk.close();
    expect((await b.sdk.searchVector("alpha"))[0]?.filepath).toBe("qmd://docs/alpha.md");
    expect(b.events).toEqual([]);
    expect(b.sdk.internal.embeddingConfig).toEqual(b.config);
  } finally { await a.sdk.close(); await b?.sdk.close(); setConfigSource(undefined); }
});
test("SDK captures a caller's inline source and recipe without mutating the original object", async () => {
  const source: { kind: "local"; path: string } = { kind: "local", path: "/fixture/original" };
  const recipe = { ...makeOnnxConfig().recipe, queryPrefix: "original: " };
  const options = { collections: {}, models: { embed: { ...makeOnnxConfig(), source, recipe } } };
  const before = JSON.stringify(options);
  const store = await createStore({ dbPath: ":memory:", config: options });
  try {
    expect(JSON.stringify(options)).toBe(before);
    source.path = "/fixture/changed"; recipe.queryPrefix = "changed: ";
    const resolved = store.internal.embeddingConfig;
    if (resolved?.backend !== "onnx") throw new Error("Missing normalized ONNX config");
    expect(resolved.source).toEqual({ kind: "local", path: "/fixture/original" });
    expect(resolved.recipe.queryPrefix).toBe("original: ");
    expect(await store.searchLex("alpha")).toEqual([]);
  } finally { await store.close(); setConfigSource(undefined); }
});
test("SDK close waits for a pending direct query and its database resolution", async () => {
  const started = deferred<void>(), release = deferred<void>();
  const f = await sdkFixture("/fixture/sdk-close", async inputs => {
    if (inputs.some(input => input.kind === "query")) { started.resolve(); await release.promise; }
  });
  let pending: Promise<unknown> | undefined, closing: Promise<void> | undefined;
  try {
    await f.sdk.embed(); pending = f.sdk.searchVector("alpha");
    const phase = await Promise.race([started.promise.then(() => "started"), pending.then(() => "settled", () => "failed")]);
    expect(phase).toBe("started");
    let closed = false; closing = f.sdk.close().then(() => { closed = true; });
    await Promise.resolve(); expect(closed).toBe(false);
    release.resolve(); expect(await pending).toMatchObject([{ filepath: "qmd://docs/alpha.md" }]);
    await closing; expect(closed).toBe(true); expect(f.events).toEqual(["embedding-close", "gguf-dispose"]);
  } finally { release.resolve(); await Promise.allSettled([...(pending ? [pending] : []), ...(closing ? [closing] : [])]); await f.sdk.close(); }
});
