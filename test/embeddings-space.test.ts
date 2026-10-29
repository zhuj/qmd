import { expect, test } from "vitest";
import { createStore } from "../src/store.js";
import { createEmbeddingStoreFixture } from "./_helpers/embedding-store.js";
import { embeddingSpaceId } from "../src/embeddings/identity.js";
import { makeSpaceDescriptor } from "./_helpers/embedding-config.js";
import { readEmbeddingSpace, planEmbeddingAdmission, commitEmbeddingSpace } from "../src/embeddings/space-store.js";
import type { EmbeddingIdentity } from "../src/embeddings/types.js";
import type { Database } from "../src/db.js";

function producer(): Extract<EmbeddingIdentity, { kind: "onnx-v1" }> {
  const descriptor = makeSpaceDescriptor(); return { kind: "onnx-v1", id: embeddingSpaceId(descriptor), descriptor };
}
function keys(db: Database) { return db.prepare("SELECT key,value FROM store_config ORDER BY key").all(); }
test("opening and reading an empty store never initializes semantic metadata", () => {
  const store = createStore(":memory:");
  try { const before = keys(store.db); expect(readEmbeddingSpace(store.db)).toEqual({ kind: "empty" }); expect(keys(store.db)).toEqual(before); }
  finally { store.close(); }
});
test("metadata commit requires a caller-owned transaction and preserves unrelated keys", () => {
  const store = createStore(":memory:");
  try {
    store.db.prepare("INSERT INTO store_config(key,value) VALUES (?,?)").run("unrelated", "keep");
    const identity = producer(), admission = planEmbeddingAdmission(readEmbeddingSpace(store.db), identity, { kind: "embed" });
    const before = keys(store.db);
    expect(() => commitEmbeddingSpace(store.db, admission)).toThrow(/transaction/i); expect(keys(store.db)).toEqual(before);
    store.db.transaction(() => commitEmbeddingSpace(store.db, admission)).immediate();
    expect(readEmbeddingSpace(store.db)).toEqual({ kind: "onnx-v1", identity });
    expect(store.db.prepare("SELECT value FROM store_config WHERE key=?").get("unrelated")).toEqual({ value: "keep" });
  } finally { store.close(); }
});
test("metadata and first fake vector batch roll back together, then commit consistently", async () => {
  const f = createEmbeddingStoreFixture(), identity = producer(), store = f.store;
  try {
    const before = keys(store.db), admission = planEmbeddingAdmission(readEmbeddingSpace(store.db), identity, { kind: "embed" });
    expect(() => store.db.transaction(() => {
      commitEmbeddingSpace(store.db, admission); store.ensureVecTable(2);
      store.insertEmbedding("fixture-alpha-beta", 0, 0, new Float32Array([1, 0]), "fake-onnx", "2026-01-01", 1, identity.id);
      throw new Error("fixture rollback");
    }).immediate()).toThrow("fixture rollback");
    expect(keys(store.db)).toEqual(before); expect(readEmbeddingSpace(store.db)).toEqual({ kind: "empty" });
    store.db.transaction(() => {
      commitEmbeddingSpace(store.db, admission); store.ensureVecTable(2);
      store.insertEmbedding("fixture-alpha-beta", 0, 0, new Float32Array([1, 0]), "fake-onnx", "2026-01-01", 1, identity.id);
    }).immediate();
    expect(readEmbeddingSpace(store.db)).toEqual({ kind: "onnx-v1", identity });
    expect(store.searchFTS("alpha", 5)).toHaveLength(1);
  } finally { await f.close(); }
});
test("stale admission cannot overwrite a concurrently initialized semantic space", () => {
  const store = createStore(":memory:");
  try {
    const first = producer(), base = first.descriptor;
    const descriptor = { ...base, recipe: { ...base.recipe, queryPrefix: "different" } };
    const second: EmbeddingIdentity = { kind: "onnx-v1", id: embeddingSpaceId(descriptor), descriptor };
    const old = planEmbeddingAdmission(readEmbeddingSpace(store.db), first, { kind: "embed" });
    const winner = planEmbeddingAdmission(readEmbeddingSpace(store.db), second, { kind: "embed" });
    store.db.transaction(() => commitEmbeddingSpace(store.db, winner)).immediate(); const before = keys(store.db);
    expect(() => store.db.transaction(() => commitEmbeddingSpace(store.db, old)).immediate()).toThrow(/changed|stale|space/i);
    expect(keys(store.db)).toEqual(before);
  } finally { store.close(); }
});
test("coherent identified legacy rows remain legacy and support stale-fingerprint full force", async () => {
  const f = createEmbeddingStoreFixture(), store = f.store;
  try {
    const before = keys(store.db);
    store.ensureVecTable(2); store.insertEmbedding("fixture-alpha-beta", 0, 0, new Float32Array([1, 0]), "old.gguf", "2026-01-01", 1, "abc123");
    const current = readEmbeddingSpace(store.db);
    expect(current).toEqual({ kind: "legacy-gguf", model: "old.gguf", fingerprint: "abc123", dimensions: 2 });
    const next: EmbeddingIdentity = { kind: "legacy-gguf", model: "new.gguf", fingerprint: "def456", dimensions: 3 };
    expect(planEmbeddingAdmission(current, next, { kind: "embed", force: true }).action).toBe("legacy-full-force");
    expect(keys(store.db)).toEqual(before);
  } finally { await f.close(); }
});
test.each([{ fingerprint: "" }, { fingerprint: "onnx-v1:" + "a".repeat(64) }, { fingerprint: "unknown" }])("unidentified or new-space rows without metadata fail closed", async ({ fingerprint }) => {
  const f = createEmbeddingStoreFixture(), store = f.store;
  try {
    store.ensureVecTable(2); store.insertEmbedding("fixture-alpha-beta", 0, 0, new Float32Array([1, 0]), "old.gguf", "2026-01-01", 1, fingerprint);
    const before = keys(store.db); expect(readEmbeddingSpace(store.db).kind).toBe("ambiguous"); expect(keys(store.db)).toEqual(before);
  } finally { await f.close(); }
});
test.each([{ corrupt: "labels" }, { corrupt: "missing-content" }, { corrupt: "rowid-map" }])("mixed/orphan vector evidence cannot qualify as legacy", async ({ corrupt }) => {
  const f = createEmbeddingStoreFixture(), store = f.store;
  try {
    store.ensureVecTable(2); store.insertEmbedding("fixture-alpha-beta", 0, 0, new Float32Array([1, 0]), "old.gguf", "2026-01-01", 1, "abc123");
    switch (corrupt) {
      case "labels": store.insertEmbedding("fixture-alpha-beta", 1, 1, new Float32Array([0, 1]), "other.gguf", "2026-01-01", 1, "def456"); break;
      case "missing-content": store.db.prepare("DELETE FROM content_vectors").run(); break;
      case "rowid-map": store.db.prepare("UPDATE vector_rows SET id=id+1000").run(); break;
      default: throw new Error("Unknown fixture case");
    }
    const before = keys(store.db); expect(readEmbeddingSpace(store.db).kind).toBe("ambiguous"); expect(keys(store.db)).toEqual(before);
  } finally { await f.close(); }
});
test.each([{ value: "not-json" }, { value: "null" }, { value: '{"version":2}' }])("malformed stored semantic metadata never repairs itself", ({ value }) => {
  const store = createStore(":memory:");
  try {
    store.db.prepare("INSERT INTO store_config(key,value) VALUES (?,?)").run("embedding_space_v1", value);
    const before = keys(store.db); expect(readEmbeddingSpace(store.db).kind).toBe("ambiguous"); expect(keys(store.db)).toEqual(before);
  } finally { store.close(); }
});
test("descriptor hash tampering, orphan runtime and read commits fail without writes", () => {
  const store = createStore(":memory:");
  try {
    const identity = producer();
    store.db.prepare("INSERT INTO store_config(key,value) VALUES (?,?)").run("embedding_runtime_v1", "{}");
    expect(readEmbeddingSpace(store.db).kind).toBe("ambiguous");
    store.db.prepare("DELETE FROM store_config WHERE key=?").run("embedding_runtime_v1");
    const read = planEmbeddingAdmission(readEmbeddingSpace(store.db), identity, { kind: "query" });
    expect(() => store.db.transaction(() => commitEmbeddingSpace(store.db, read)).immediate()).toThrow(/read/i);
    store.db.prepare("INSERT INTO store_config(key,value) VALUES (?,?)").run("embedding_space_v1",
      JSON.stringify({ version: 1, id: "onnx-v1:" + "0".repeat(64), descriptor: identity.descriptor }));
    const before = keys(store.db); expect(readEmbeddingSpace(store.db).kind).toBe("ambiguous"); expect(keys(store.db)).toEqual(before);
  } finally { store.close(); }
});
test("an incompatible empty physical dimension or invalid provenance cannot partially initialize metadata", () => {
  const store = createStore(":memory:");
  try {
    const admission = planEmbeddingAdmission(readEmbeddingSpace(store.db), producer(), { kind: "embed" });
    const before = keys(store.db);
    expect(() => store.db.transaction(() => commitEmbeddingSpace(store.db, admission, {
      requestedDevice: "cuda", initializedMode: "cpu", graphNodePlacement: "unknown", cpuRetryReason: "unclassified-private-detail",
    })).immediate()).toThrow(); expect(keys(store.db)).toEqual(before);
    store.ensureVecTable(3);
    expect(() => store.db.transaction(() => commitEmbeddingSpace(store.db, admission)).immediate()).toThrow(/dimension|separate/i);
    expect(keys(store.db)).toEqual(before);
  } finally { store.close(); }
});
