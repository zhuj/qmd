import { describe, test, expect } from "vitest";
import { generateEmbeddings } from "../src/store.js";
import { fixtureEmbeddingConfig, createEmbeddingStoreFixture } from "./_helpers/embedding-store.js";
import type { EmbeddingCapability } from "../src/embeddings/types.js";
import type { Database } from "../src/db.js";

function captureState(db: Database) {
  return {
    sqliteMaster: db.prepare("SELECT name, sql FROM sqlite_master WHERE type='table' OR type='virtualtable' ORDER BY name").all(),
    contentVectors: db.prepare("SELECT * FROM content_vectors").all(),
    vectorRows: db.prepare("SELECT * FROM vector_rows ORDER BY id").all(),
    storeConfig: db.prepare("SELECT * FROM store_config").all(),
  };
}

describe("onnx-force-compat regressions", () => {
  test("case 1: configurable ONNX store with protected fake vector; force with wrong model rejects before state change", async () => {
    const fixture = createEmbeddingStoreFixture(), store = fixture.store;
    try {
      store.ensureVecTable(2);
      store.insertEmbedding("fixture-alpha-beta", 0, 0, new Float32Array([1, 0]), "protected-label", "2026-01-01", 1, "abc123");
      const before = captureState(store.db);
      const caught: unknown = await generateEmbeddings(store, { force: true, model: "not-the-selected-producer" }).catch((error: unknown) => error);
      const after = captureState(store.db);
      expect(after.contentVectors).toEqual(before.contentVectors);
      expect(after.vectorRows).toEqual(before.vectorRows);
      expect(after.sqliteMaster).toEqual(before.sqliteMaster);
      expect(after.storeConfig).toEqual(before.storeConfig);
      expect(caught).toBeInstanceOf(Error);
      if (!(caught instanceof Error)) throw new Error("Missing model conflict");
      expect(caught.message).toMatch(/producer|model|override/i);
      expect(fixture.events.some(e => e.startsWith("gguf-"))).toBe(false);
    } finally {
      await fixture.close();
    }
  });

  test("case 2: explicit fakeEmbeddingCapability with preflight failure; force rejects after prepare 1 before embed/clear", async () => {
    const config = fixtureEmbeddingConfig();
    let prepareCalls = 0;
    let embedCalls = 0;
    const cap: EmbeddingCapability = {
      config,
      modelLabel: "fixture-onnx",
      provenance: undefined,
      prepare: async () => { prepareCalls += 1; throw new Error("fixture preflight failure"); },
      encode: async (text: string) => Array.from(text, c => c.codePointAt(0) ?? 0),
      decode: async (tokens: readonly number[]) => String.fromCodePoint(...tokens),
      embed: async () => { embedCalls += 1; return [new Float32Array([0.1, 0.2])]; },
      close: async () => {},
    };
    const fixture = createEmbeddingStoreFixture(config, cap);
    const db = fixture.store.db;
    try {
      fixture.store.ensureVecTable(2);
      fixture.store.insertEmbedding("fixture-alpha-beta", 0, 0, new Float32Array([1, 0]), "protected-label", "2026-01-01", 1, "abc123");
      const before = captureState(db);
      const caught: unknown = await generateEmbeddings(fixture.store, { force: true }).catch((error: unknown) => error);
      const after = captureState(db);
      expect(after.contentVectors).toEqual(before.contentVectors);
      expect(after.vectorRows).toEqual(before.vectorRows);
      expect(after.sqliteMaster).toEqual(before.sqliteMaster);
      expect(after.storeConfig).toEqual(before.storeConfig);
      expect(caught).toEqual(new Error("fixture preflight failure"));
      expect(prepareCalls).toBe(1);
      expect(embedCalls).toBe(0);
      expect(fixture.events.some(e => e === "embed")).toBe(false);
      expect(fixture.events.some(e => e.startsWith("gguf-"))).toBe(false);
    } finally {
      await fixture.close();
    }
  });
});
