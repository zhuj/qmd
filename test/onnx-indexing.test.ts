import { expect, test } from "vitest";
import { generateEmbeddings } from "../src/store.js";
import { readEmbeddingSpace } from "../src/embeddings/space-store.js";
import { createEmbeddingStoreFixture } from "./_helpers/embedding-store.js";
import { controlledIndexingFixture } from "./_helpers/indexing-store.js";
import { createArtifactManifest } from "../src/embeddings/identity.js";

test("configured ONNX indexing persists its actual identity, reuses it and never calls GGUF", async () => {
  const f = createEmbeddingStoreFixture();
  try {
    const result = await generateEmbeddings(f.store);
    expect(result.docsProcessed).toBe(1); expect(result.chunksEmbedded).toBe(1); expect(result.errors).toBe(0);
    const space = readEmbeddingSpace(f.store.db);
    expect(space.kind).toBe("onnx-v1");
    if (space.kind !== "onnx-v1") throw new Error("Missing stored ONNX space");
    expect(f.store.db.prepare("SELECT embed_fingerprint,model FROM content_vectors").all())
      .toEqual([{ embed_fingerprint: space.identity.id, model: f.store.embedding?.modelLabel }]);
    const embedded = f.inputs.length;
    const again = await generateEmbeddings(f.store);
    expect(again.docsProcessed).toBe(0); expect(again.chunksEmbedded).toBe(0); expect(f.inputs).toHaveLength(embedded);
    const preparations = f.events.filter(event => event === "prepare").length;
    expect((await generateEmbeddings(f.store, { force: true })).chunksEmbedded).toBe(1);
    expect(f.events.filter(event => event === "prepare").length).toBe(preparations + 1);
    expect(readEmbeddingSpace(f.store.db)).toEqual(space);
    expect(f.events.some(event => event.startsWith("gguf-"))).toBe(false);
  } finally { await f.close(); }
});
test("a same-dimensional changed producer cannot use force to replace an admitted ONNX space", async () => {
  const f = controlledIndexingFixture();
  try {
    await generateEmbeddings(f.store);
    const before = {
      vectors: f.store.db.prepare("SELECT * FROM content_vectors ORDER BY seq").all(),
      rows: f.store.db.prepare("SELECT * FROM vector_rows ORDER BY id").all(),
      metadata: f.store.db.prepare("SELECT * FROM store_config ORDER BY key").all(),
      layout: f.store.db.prepare("SELECT sql FROM sqlite_master WHERE name='vectors_by_collection'").all(),
    };
    const embedded = f.calls.length;
    f.descriptor.artifactManifest = createArtifactManifest(f.descriptor.artifactManifest.source,
      f.descriptor.artifactManifest.files.map(file => file.path.endsWith(".onnx") ? { ...file, sha256: "b".repeat(64) } : file));
    await expect(generateEmbeddings(f.store, { force: true })).rejects.toThrow(/space|database|incompatible/i);
    expect(f.store.db.prepare("SELECT * FROM content_vectors ORDER BY seq").all()).toEqual(before.vectors);
    expect(f.store.db.prepare("SELECT * FROM vector_rows ORDER BY id").all()).toEqual(before.rows);
    expect(f.store.db.prepare("SELECT * FROM store_config ORDER BY key").all()).toEqual(before.metadata);
    expect(f.store.db.prepare("SELECT sql FROM sqlite_master WHERE name='vectors_by_collection'").all()).toEqual(before.layout);
    expect(f.calls).toHaveLength(embedded);
  } finally { await f.close(); }
});
