import { expect, test } from "vitest";
import { generateEmbeddings, hybridQuery, searchVec, structuredSearch, vectorSearchQuery } from "../src/store.js";
import { createEmbeddingStoreFixture } from "./_helpers/embedding-store.js";
import { controlledIndexingFixture } from "./_helpers/indexing-store.js";
import { deferred } from "./_helpers/deferred.js";

test("store-bound direct vectors and supplied structured vec/hyde texts use the indexed producer", async () => {
  const f = createEmbeddingStoreFixture();
  try {
    await generateEmbeddings(f.store);
    const producer = f.store.embedding;
    if (!producer) throw new Error("Missing configured fixture producer");
    f.inputs.length = 0; f.events.length = 0;
    expect((await f.store.searchVec("alpha", producer.modelLabel)).map(result => result.filepath)).toEqual(["qmd://docs/alpha.md"]);
    expect(f.inputs).toEqual([{ kind: "query", text: "alpha" }]);
    f.inputs.length = 0;
    const supplied = [{ type: "vec", query: "alpha" }, { type: "hyde", query: "beta" }];
    const results = await structuredSearch(f.store, [{ type: "vec", query: "alpha" }, { type: "hyde", query: "beta" }]);
    expect(results.map(result => result.file)).toEqual(["qmd://docs/alpha.md"]);
    expect(f.inputs).toEqual(supplied.map(input => ({ kind: "query", text: input.query })));
    expect(f.events.some(event => event.startsWith("gguf-"))).toBe(false);
  } finally { await f.close(); }
});
test("legacy positional precomputed vectors cannot bypass an ONNX space without its identity/context", async () => {
  const f = createEmbeddingStoreFixture();
  try {
    await generateEmbeddings(f.store); f.inputs.length = 0; f.events.length = 0;
    await expect(searchVec(f.store.db, "alpha", "unrelated.gguf", 20, undefined, undefined, [0.6, 0.8]))
      .rejects.toThrow(/identity|context|producer|ONNX/i);
    expect(f.inputs).toEqual([]); expect(f.events).toEqual([]);
  } finally { await f.close(); }
});
test("CLI vector wrapper uses the configured CPU producer and original query without expansion", async () => {
  const f = createEmbeddingStoreFixture();
  try {
    await generateEmbeddings(f.store); f.inputs.length = 0; f.events.length = 0;
    const results = await vectorSearchQuery(f.store, "alpha", { hooks: { onExpand: () => { throw new Error("forbidden expansion hook"); } } });
    expect(results.map(result => result.file)).toEqual(["qmd://docs/alpha.md"]);
    expect(f.inputs).toEqual([{ kind: "query", text: "alpha" }]);
    expect(f.events.some(event => event.startsWith("gguf-"))).toBe(false);
  } finally { await f.close(); }
});
test("bound precomputed vectors require exact identity, dimension and finite cosine data", async () => {
  const f = createEmbeddingStoreFixture();
  try {
    await generateEmbeddings(f.store);
    const cap = f.store.embedding;
    if (!cap) throw new Error("Missing fixture producer");
    const identity = await cap.prepare();
    if (identity.kind !== "onnx-v1") throw new Error("Wrong fixture identity");
    f.inputs.length = 0;
    await expect(f.store.searchVec("alpha", cap.modelLabel, 20, undefined, undefined, [0.6, 0.8])).rejects.toThrow(/identity/i);
    await expect(f.store.searchVec("alpha", cap.modelLabel, 20, undefined, undefined, [0.6, 0.8], undefined, "onnx-v1:wrong")).rejects.toThrow(/identity/i);
    for (const vector of [[1], [NaN, 1], [0, 0]]) {
      await expect(f.store.searchVec("alpha", cap.modelLabel, 20, undefined, undefined, vector, undefined, identity.id)).rejects.toThrow();
    }
    expect((await f.store.searchVec("alpha", cap.modelLabel, 20, undefined, undefined, [0.6, 0.8], undefined, identity.id))[0]?.filepath).toBe("qmd://docs/alpha.md");
    expect(f.inputs).toEqual([]);
  } finally { await f.close(); }
});
test("query overflow and a conflicting bound model reject before query embedding while lexical retrieval survives", async () => {
  const f = createEmbeddingStoreFixture();
  try {
    await generateEmbeddings(f.store);
    const cap = f.store.embedding;
    if (!cap) throw new Error("Missing fixture producer");
    f.inputs.length = 0; f.events.length = 0;
    await expect(f.store.searchVec("alpha", "other.gguf")).rejects.toThrow(/model|producer/i);
    expect(f.events).toEqual([]);
    await expect(f.store.searchVec("alpha".repeat(100), cap.modelLabel)).rejects.toThrow(/shorten|limit|token|budget/i);
    expect(f.inputs).toEqual([]);
    expect(f.store.searchFTS("alpha").map(result => result.filepath)).toEqual(["qmd://docs/alpha.md"]);
    expect(f.events.some(event => event.startsWith("gguf-"))).toBe(false);
  } finally { await f.close(); }
});
test("direct query holds the database lease through post-inference admission and vector resolution", async () => {
  const started = deferred<void>(), release = deferred<void>();
  const f = controlledIndexingFixture(async inputs => {
    if (inputs.some(input => input.kind === "query")) { started.resolve(); await release.promise; }
  });
  let pending: Promise<unknown> | undefined, closing: Promise<void> | undefined;
  try {
    await generateEmbeddings(f.store);
    const cap = f.store.embedding;
    if (!cap) throw new Error("Missing fixture producer");
    pending = f.store.searchVec("alpha", cap.modelLabel);
    const phase = await Promise.race([started.promise.then(() => "started"), pending.then(() => "settled", () => "failed")]);
    expect(phase).toBe("started");
    let closed = false;
    closing = f.store.closeAsync().then(() => { closed = true; });
    await Promise.resolve(); expect(closed).toBe(false);
    release.resolve();
    expect(await pending).toMatchObject([{ filepath: "qmd://docs/alpha.md" }]);
    await closing; expect(closed).toBe(true);
  } finally { release.resolve(); await Promise.allSettled([...(pending ? [pending] : []), ...(closing ? [closing] : [])]); await f.close(); }
});
test("a changed same-dimensional producer cannot query indexed vectors but lexical search remains usable", async () => {
  const f = controlledIndexingFixture();
  try {
    await generateEmbeddings(f.store);
    const cap = f.store.embedding;
    if (!cap) throw new Error("Missing fixture producer");
    f.descriptor.recipe = { ...f.descriptor.recipe, queryPrefix: "changed:" };
    const calls = f.calls.length;
    await expect(f.store.searchVec("alpha", cap.modelLabel)).rejects.toThrow(/separate|space|incompatible/i);
    await expect(structuredSearch(f.store, [{ type: "vec", query: "alpha" }])).rejects.toThrow(/separate|space|incompatible/i);
    expect(f.calls.length).toBe(calls);
    expect(f.store.searchFTS("alpha").map(result => result.filepath)).toEqual(["qmd://docs/alpha.md"]);
    expect(f.events.some(event => event.startsWith("gguf-"))).toBe(false);
  } finally { await f.close(); }
});
test("collection/metadata restrictions and existing fusion explanations survive all configured query paths", async () => {
  const f = createEmbeddingStoreFixture();
  try {
    await generateEmbeddings(f.store);
    const cap = f.store.embedding;
    if (!cap) throw new Error("Missing fixture producer");
    expect(await f.store.searchVec("alpha", cap.modelLabel, 20, "other")).toEqual([]);
    expect(await hybridQuery(f.store, "alpha", { collection: "other" })).toEqual([]);
    expect(await structuredSearch(f.store, [{ type: "vec", query: "alpha" }], { collections: ["other"] })).toEqual([]);
    const filtered = { field: "status", operator: "eq", value: "published" };
    expect(await f.store.searchVec("alpha", cap.modelLabel, 20, undefined, undefined, undefined, { field: "status", operator: "eq", value: "published" })).toEqual([]);
    expect(await hybridQuery(f.store, "alpha", { filter: { field: filtered.field, operator: "eq", value: filtered.value } })).toEqual([]);
    expect(await structuredSearch(f.store, [{ type: "vec", query: "alpha" }], { filter: { field: filtered.field, operator: "eq", value: filtered.value } })).toEqual([]);
    const first = await hybridQuery(f.store, "alpha", { explain: true });
    expect(await hybridQuery(f.store, "alpha", { explain: true })).toEqual(first);
    expect(first.map(result => result.file)).toEqual(["qmd://docs/alpha.md"]);
    expect(first[0]?.explain?.rrf.contributions.map(item => item.source).sort()).toEqual(["fts", "vec"]);
    expect(f.events.some(event => event.startsWith("gguf-"))).toBe(false);
  } finally { await f.close(); }
});
