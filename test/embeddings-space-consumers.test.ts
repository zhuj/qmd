import { expect, test } from "vitest";
import { copyVectorsToNewCollections, getHashesNeedingEmbedding, getHashesNeedingEmbeddingInSpace, getIndexHealth, getEmbeddingFingerprint, upsertStoreCollection } from "../src/store.js";
import { readEmbeddingSpace, planEmbeddingAdmission, commitEmbeddingSpace } from "../src/embeddings/space-store.js";
import { embeddingSpaceId } from "../src/embeddings/identity.js";
import { makeSpaceDescriptor } from "./_helpers/embedding-config.js";
import { createEmbeddingStoreFixture } from "./_helpers/embedding-store.js";
import type { EmbeddingIdentity } from "../src/embeddings/types.js";

function producer(prefix = ""): Extract<EmbeddingIdentity, { kind: "onnx-v1" }> {
  const base = makeSpaceDescriptor(), descriptor = { ...base, recipe: { ...base.recipe, queryPrefix: prefix } };
  return { kind: "onnx-v1", id: embeddingSpaceId(descriptor), descriptor };
}
test("health/completeness use full semantic fingerprint rather than a six-character model label", async () => {
  const f = createEmbeddingStoreFixture(), identity = producer();
  try {
    expect(getHashesNeedingEmbeddingInSpace(f.store.db, identity)).toBe(1);
    const admission = planEmbeddingAdmission(readEmbeddingSpace(f.store.db), identity, { kind: "embed" });
    f.store.db.transaction(() => {
      commitEmbeddingSpace(f.store.db, admission); f.store.ensureVecTable(2);
      f.store.insertEmbedding("fixture-alpha-beta", 0, 0, new Float32Array([1, 0]), "actual-display-label", "2026-01-01", 1, identity.id);
    }).immediate();
    expect(getHashesNeedingEmbeddingInSpace(f.store.db, identity)).toBe(0);
    expect(getIndexHealth(f.store.db, "different-display-label", identity).needsEmbedding).toBe(0);
    expect(f.store.getHashesNeedingEmbedding("another-display-label", identity)).toBe(0);
    expect(f.store.getIndexHealth(undefined, identity).needsEmbedding).toBe(0);
    expect(() => getHashesNeedingEmbeddingInSpace(f.store.db, producer("wrong"))).toThrow(/space|separate/i);
  } finally { await f.close(); }
});
test("copy validates semantic space before mutation and copies only the admitted identity", async () => {
  const f = createEmbeddingStoreFixture(), identity = producer(), db = f.store.db;
  try {
    const admission = planEmbeddingAdmission(readEmbeddingSpace(db), identity, { kind: "embed" });
    db.transaction(() => {
      commitEmbeddingSpace(db, admission); f.store.ensureVecTable(2);
      f.store.insertEmbedding("fixture-alpha-beta", 0, 0, new Float32Array([1, 0]), "actual-display-label", "2026-01-01", 1, identity.id);
    }).immediate();
    upsertStoreCollection(db, "other", { path: "/fixture/other", pattern: "**/*.md" });
    f.store.insertDocument("other", "same.md", "Alpha", "fixture-alpha-beta", "2026-01-01", "2026-01-01");
    const before = db.prepare("SELECT * FROM vector_rows ORDER BY id").all();
    expect(() => copyVectorsToNewCollections(db, undefined, producer("wrong"))).toThrow(/space|separate/i);
    expect(db.prepare("SELECT * FROM vector_rows ORDER BY id").all()).toEqual(before);
    expect(copyVectorsToNewCollections(db, undefined, identity)).toEqual({ copied: 1, queued: 0 });
    expect(db.prepare("SELECT * FROM vector_rows").all()).toHaveLength(2);
    expect(f.store.searchFTS("alpha", 5)).toHaveLength(2);
  } finally { await f.close(); }
});
test.each([{ gap: true }, { gap: false }])("completeness rejects missing sequence positions or inconsistent total counts", async ({ gap }) => {
  const f = createEmbeddingStoreFixture(), identity = producer(), db = f.store.db;
  try {
    const admission = planEmbeddingAdmission(readEmbeddingSpace(db), identity, { kind: "embed" });
    db.transaction(() => {
      commitEmbeddingSpace(db, admission); f.store.ensureVecTable(2);
      f.store.insertEmbedding("fixture-alpha-beta", 0, 0, new Float32Array([1, 0]), "label", "2026-01-01", 2, identity.id);
      f.store.insertEmbedding("fixture-alpha-beta", gap ? 2 : 1, 1, new Float32Array([0, 1]), "label", "2026-01-01", gap ? 2 : 1, identity.id);
    }).immediate();
    expect(getHashesNeedingEmbeddingInSpace(db, identity)).toBe(1);
  } finally { await f.close(); }
});
test("legacy string-only completeness and health signatures retain existing behavior", async () => {
  const model = "old.gguf", f = createEmbeddingStoreFixture({ backend: "gguf", model }), fingerprint = getEmbeddingFingerprint(model);
  try {
    expect(getHashesNeedingEmbedding(f.store.db, undefined, model)).toBe(1);
    f.store.ensureVecTable(2);
    f.store.insertEmbedding("fixture-alpha-beta", 0, 0, new Float32Array([1, 0]), model, "2026-01-01", 1, fingerprint);
    expect(getHashesNeedingEmbedding(f.store.db, undefined, model)).toBe(0);
    expect(f.store.getHashesNeedingEmbedding(model)).toBe(0);
    expect(f.store.getIndexHealth(model).needsEmbedding).toBe(0);
  } finally { await f.close(); }
});
