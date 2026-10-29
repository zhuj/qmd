import { expect, test } from "vitest";
import { generateEmbeddings, hybridQuery, structuredSearch, configureStoreEmbeddings, createStore, upsertStoreCollection } from "../src/store.js";
import { createEmbeddingStoreFixture, ForbiddenLlama, fixtureEmbeddingConfig } from "./_helpers/embedding-store.js";
import { makeOnnxConfig, makeSpaceDescriptor } from "./_helpers/embedding-config.js";
import { resolveEmbeddingConfig } from "../src/embeddings/config.js";
import { createArtifactManifest, embeddingSpaceId } from "../src/embeddings/identity.js";
import { formatEmbeddingInput } from "../src/embeddings/recipe.js";
import { planEmbeddingAdmission, readEmbeddingSpace, commitEmbeddingSpace } from "../src/embeddings/space-store.js";
import type { EmbeddingCapability, EmbeddingIdentity, EmbeddingSpaceDescriptor } from "../src/embeddings/types.js";

test("index, direct, hybrid and structured share one ONNX producer without GGUF", async () => {
  const f = createEmbeddingStoreFixture();
  try {
    const embedded = await generateEmbeddings(f.store);
    expect(embedded.chunksEmbedded).toBeGreaterThan(0);
    const label = f.store.embedding?.modelLabel; if (!label) throw new Error("fixture producer");
    await f.store.searchVec("alpha", label); await hybridQuery(f.store, "alpha");
    await structuredSearch(f.store, [{ type: "lex", query: "alpha" }, { type: "vec", query: "alpha" }]);
    expect(f.events.some(e => e.startsWith("gguf-"))).toBe(false);
    expect(f.store.db.prepare("SELECT COUNT(DISTINCT embed_fingerprint) AS n FROM content_vectors").get()).toEqual({ n: 1 });
  } finally { await f.close(); }
});

test("same-dimension descriptor changed identity rejected by production admission before embed", async () => {
  const config = fixtureEmbeddingConfig();
  const base = makeSpaceDescriptor();
  let descriptor: EmbeddingSpaceDescriptor = {
    ...base, artifactManifest: createArtifactManifest(config.source, base.artifactManifest.files), recipe: config.recipe,
  };
  let prepareCalls = 0, embedCalls = 0;
  const tokenizer = { encode: async (text: string, special?: boolean) => special ? [0, ...Array.from(text, c => c.codePointAt(0) ?? 0), 0] : Array.from(text, c => c.codePointAt(0) ?? 0), decode: async (tokens: readonly number[]) => String.fromCodePoint(...tokens) };
  const raw: EmbeddingCapability = {
    config, modelLabel: "changed-producer", provenance: undefined,
    prepare: async () => { prepareCalls++; return { kind: "onnx-v1", id: embeddingSpaceId(descriptor), descriptor }; },
    encode: text => tokenizer.encode(text, true),
    decode: tokens => tokenizer.decode(tokens),
    embed: async values => { embedCalls++; return values.map(() => new Float32Array([0.6, 0.8])); },
    close: async () => {},
    measureTokens: async (input, signal) => {
      signal?.throwIfAborted();
      const text = await tokenizer.encode(input.text, false);
      const formatted = await formatEmbeddingInput(input, config.recipe, tokenizer);
      const total = await tokenizer.encode(formatted, true);
      return { text: text.length, total: total.length };
    },
  };
  const f = createEmbeddingStoreFixture(config, raw);
  try {
    const embedded = await generateEmbeddings(f.store);
    expect(embedded.chunksEmbedded).toBeGreaterThan(0);
    const db = f.store.db;
    const vecBefore = db.prepare("SELECT hash,seq,pos,model,embed_fingerprint,total_chunks,embedded_at FROM content_vectors ORDER BY hash,seq").all();
    expect(vecBefore.length).toBeGreaterThan(0);
    const payloadSnapshot = JSON.stringify(vecBefore);
    expect(payloadSnapshot.length).toBeGreaterThan(0);
    const mapBefore = db.prepare("SELECT id,hash,seq,collection_id FROM vector_rows ORDER BY id").all();
    const physBefore = db.prepare("SELECT rowid,collection_id,hex(embedding) AS embedding_hex FROM vectors_by_collection ORDER BY rowid").all();
    const spaceBefore = db.prepare("SELECT value FROM store_config WHERE key='embedding_space_v1'").get<{ value: string }>();
    if (!spaceBefore) throw new Error("Initial embedding space was not committed");
    const runtimeBefore = db.prepare("SELECT * FROM store_config ORDER BY key").all();
    const preparedBeforeMismatch = prepareCalls;
    const embeddedBeforeMismatch = embedCalls;
    descriptor = { ...descriptor, recipe: { ...descriptor.recipe, queryPrefix: "changed-" } };
    expect(embeddingSpaceId(descriptor)).not.toBe(JSON.parse(String(spaceBefore.value)).id);
    let rejected = false;
    try { await generateEmbeddings(f.store, { force: true }); } catch (e: unknown) { rejected = /Embedding space changed|incompatible|space|mismatch/i.test(String(e)); }
    expect(rejected).toBe(true);
    expect(prepareCalls).toBeGreaterThan(preparedBeforeMismatch);
    expect(embedCalls).toBe(embeddedBeforeMismatch);
    const vecAfter = db.prepare("SELECT hash,seq,pos,model,embed_fingerprint,total_chunks,embedded_at FROM content_vectors ORDER BY hash,seq").all();
    expect(JSON.stringify(vecAfter)).toBe(JSON.stringify(vecBefore));
    expect(JSON.stringify(db.prepare("SELECT id,hash,seq,collection_id FROM vector_rows ORDER BY id").all())).toBe(JSON.stringify(mapBefore));
    expect(JSON.stringify(db.prepare("SELECT rowid,collection_id,hex(embedding) AS embedding_hex FROM vectors_by_collection ORDER BY rowid").all())).toBe(JSON.stringify(physBefore));
    expect(JSON.stringify(db.prepare("SELECT value FROM store_config WHERE key='embedding_space_v1'").get())).toBe(JSON.stringify(spaceBefore));
    expect(JSON.stringify(db.prepare("SELECT * FROM store_config ORDER BY key").all())).toBe(JSON.stringify(runtimeBefore));
    const lex = await f.store.searchFTS("alpha");
    expect(Array.isArray(lex)).toBe(true); expect(lex.length).toBeGreaterThan(0);
  } finally { await f.close(); }
});

test("failed preflight preserves DB and lexical stays usable", async () => {
  const s = createStore(":memory:");
  try {
    s.llm = new ForbiddenLlama([]);
    upsertStoreCollection(s.db, "docs", { path: "/fixture/docs", pattern: "**/*.md" });
    s.insertContent("preflight-hash", "alpha", "2026-01-01");
    s.insertDocument("docs", "preflight.md", "Preflight", "preflight-hash", "2026-01-01", "2026-01-01");
    const badCap: EmbeddingCapability = {
      config: resolveEmbeddingConfig(makeOnnxConfig(), { defaultGgufModel: "unused", hostParallelism: 8 }),
      modelLabel: "fail-preflight", provenance: undefined,
      prepare: async () => { throw new Error("preflight failure"); },
      encode: async t => [], decode: async () => "", embed: async () => [], close: async () => {},
    };
    let prepared = 0;
    const tracked: EmbeddingCapability = {
      ...badCap,
      prepare: async () => { prepared++; throw new Error("preflight failure"); },
    };
    configureStoreEmbeddings(s, tracked.config, tracked);
    const before = s.db.prepare("SELECT COUNT(*) AS c FROM content_vectors").get<{ c: number }>();
    if (!before) throw new Error("Missing initial vector count");
    expect(before.c).toBe(0);
    let failedPrepare = false;
    try { await generateEmbeddings(s, { maxDurationMs: 500 }); expect(false).toBe(true); } catch (e: any) {
      failedPrepare = prepared > 0 && /preflight failure/i.test(String(e));
    }
    expect(failedPrepare).toBe(true); // production generateEmbeddings reached and failed the preflight
    const after = s.db.prepare("SELECT COUNT(*) AS c FROM content_vectors").get<{ c: number }>();
    if (!after) throw new Error("Missing vector count after preflight failure");
    expect(after.c).toBe(0); // the failed preflight commits nothing
    const lex = await s.searchFTS("alpha"); // lexical usable on the SAME store
    expect(Array.isArray(lex)).toBe(true);
    expect(lex.length).toBeGreaterThan(0);
  } finally { await s.closeAsync(); }
});

test("cancelled batch cannot commit (abort lands after embed resolves)", async () => {
  const f = createEmbeddingStoreFixture();
  try {
    const cap = f.store.embedding!;
    let embedCalls = 0;
    const controller = new AbortController();
    const origEmbed = cap.embed.bind(cap);
    const wrappedEmbed = async (vals: any[], signal?: AbortSignal) => {
      embedCalls++;
      const out = (await origEmbed(vals, signal)) as Float32Array[];
      controller.abort(); //Cancellation lands after the embed resolves, before persistence
      return out;
    };
    const wrapped: EmbeddingCapability = {
      ...cap,
      embed: wrappedEmbed,
      prepare: cap.prepare.bind(cap), encode: cap.encode.bind(cap), decode: cap.decode.bind(cap), close: cap.close.bind(cap),
    };
    const s = createStore(":memory:");
    try {
      s.llm = new ForbiddenLlama(f.events);
      upsertStoreCollection(s.db, "docs", { path: "/fixture/docs", pattern: "**/*.md" });
      const date = "2026-01-01T00:00:00Z";
      s.insertContent("fixture-cancel-hash", "alpha beta", date);
      s.insertDocument("docs", "cancel.md", "Cancel", "fixture-cancel-hash", date, date);
      configureStoreEmbeddings(s, wrapped.config, wrapped);
      const before = s.db.prepare("SELECT COUNT(*) AS c FROM content_vectors").get<{ c: number }>();
      if (!before) throw new Error("Missing initial vector count");
      let aborted = false;
      try { await generateEmbeddings(s, { signal: controller.signal, maxDurationMs: 500 }); } catch (e: any) {
        aborted = /abort/i.test(String(e)) || e?.name === "AbortError";
      }
      expect(aborted).toBe(true); // budget.signal is composed via AbortSignal.any, so the abort propagates
      expect(embedCalls).toBe(1); // first embed reached; pipeline severed at the batch boundary
      const after = s.db.prepare("SELECT COUNT(*) AS c FROM content_vectors").get<{ c: number }>();
      if (!after) throw new Error("Missing vector count after cancellation");
      expect(after.c).toBe(before.c); // persistBatch aborts before its transaction: cancelled batch never commits
    } finally { await s.closeAsync(); }
  } finally { await f.close(); }
});

test("two stores with one closing: isolation and real work preserved", async () => {
  const a = createEmbeddingStoreFixture(); const b = createEmbeddingStoreFixture();
  try {
    // Seed and embed b BEFORE closing a (so b has vector index and content_vectors)
    await generateEmbeddings(b.store);
    await a.store.embedding!.prepare(); await b.store.embedding!.prepare();
    await a.close();
    // b does real vector-backed query after a closes (hasVectorIndex true, content_vectors non-empty)
    const result = await hybridQuery(b.store, "alpha");
    expect(Array.isArray(result)).toBe(true);
    expect(result.length).toBeGreaterThan(0); // seeded "alpha" corpus hit, not an empty fallback
    // b's raw capability events: no "close" until its own shutdown
    expect(b.events.some(e => e === "close")).toBe(false);
    await b.store.embedding!.close();
    await b.close();
    expect(b.events.some(e => e === "close")).toBe(true);
  } finally { await a.close().catch(() => {}); await b.close().catch(() => {}); }
});

test("entry mapping T14/T15: descriptor identity tied to pipeline", async () => {
  const f = createEmbeddingStoreFixture();
  try {
    if (!f.store.embedding) throw new Error("no embedding");
    const identity = await f.store.embedding.prepare();
    expect(identity.kind === "onnx-v1" ? identity.id : identity.fingerprint).toBeTruthy();
    f.store.db.transaction(() => commitEmbeddingSpace(f.store.db, planEmbeddingAdmission(readEmbeddingSpace(f.store.db), identity, { kind: "embed" }))).immediate();
    const row = f.store.db.prepare("SELECT value FROM store_config WHERE key='embedding_space_v1'").get<{ value: string }>();
    if (!row) throw new Error("Missing committed embedding space");
    expect(row && typeof row.value === "string").toBe(true);
    const parsed = JSON.parse(row.value);
    expect(parsed && (parsed.backend === "onnx" || parsed.kind === "onnx-v1" || parsed.descriptor?.backend === "onnx")).toBe(true);
    const onnxId = identity.kind === "onnx-v1" ? String(identity.id) : String(identity.fingerprint);
    expect(String(row.value)).toContain(onnxId); // the persisted row is THIS producer's identity, not just any onnx shape
    if (identity.kind === "onnx-v1") {
      expect(parsed).toEqual({ version: 1, id: identity.id, descriptor: identity.descriptor }); // exact persisted identity+descriptor
    }
  } finally { await f.close(); }
});
