import { expect, test } from "vitest";
import type { Token } from "node-llama-cpp";
import { LlamaCpp, type EmbeddingResult } from "../src/llm.js";
import { createStore, generateEmbeddings, insertContent, insertDocument, upsertStoreCollection } from "../src/store.js";

class FixtureGguf extends LlamaCpp {
  calls = 0;
  constructor(private readonly dimensions: number) { super({ embedModel: "new.gguf", inactivityTimeoutMs: 0 }); }
  override async tokenize(text: string): Promise<Token[]> {
    // Synthetic IDs cross only the native library's opaque numeric brand; no native tokenizer is loaded.
    return Array.from(text, value => value.codePointAt(0) ?? 0) as Token[];
  }
  override async detokenize(tokens: Token[]): Promise<string> { return String.fromCodePoint(...tokens); }
  override async embed(): Promise<EmbeddingResult> {
    this.calls++;
    return { model: this.embedModelName, embedding: Array.from({ length: this.dimensions }, (_, index) => index === 0 ? 1 : 0) };
  }
  override async embedBatch(texts: string[]): Promise<EmbeddingResult[]> { return Promise.all(texts.map(() => this.embed())); }
}
function fixture(dimensions: number) {
  const store = createStore(":memory:"), llm = new FixtureGguf(dimensions);
  try {
    store.llm = llm;
    upsertStoreCollection(store.db, "docs", { path: "/fixture/docs", pattern: "**/*.md" });
    insertContent(store.db, "fixture-doc", "alpha beta", "2026-01-01");
    insertDocument(store.db, "docs", "alpha.md", "alpha", "fixture-doc", "2026-01-01", "2026-01-01");
    store.ensureVecTable(2);
    store.insertEmbedding("fixture-doc", 0, 0, new Float32Array([1, 0]), "old.gguf", "2026-01-01", 1, "abc123");
    return { store, llm };
  } catch (error) { store.close(); throw error; }
}
test.each([{ dimensions: 2 }, { dimensions: 3 }])("identified legacy full force uses the actual new $dimensions-dimensional producer", async ({ dimensions }) => {
  const f = fixture(dimensions);
  try {
    const result = await generateEmbeddings(f.store, { force: true });
    expect(result.errors).toBe(0); expect(result.chunksEmbedded).toBe(1);
    expect(f.store.db.prepare("SELECT model FROM content_vectors").all()).toEqual([{ model: "new.gguf" }]);
    expect(f.store.db.prepare("SELECT length(embedding)/4 AS dimensions FROM vectors_by_collection").all()).toEqual([{ dimensions }]);
    const calls = f.llm.calls;
    await generateEmbeddings(f.store, { force: true });
    expect(f.llm.calls).toBeGreaterThan(calls);
    expect(f.store.embedding).toBeUndefined();
  } finally { f.store.close(); }
});
test("scoped legacy replacement leaves the old index and metadata unchanged", async () => {
  const f = fixture(3);
  try {
    const before = f.store.db.prepare("SELECT * FROM content_vectors").all();
    await expect(generateEmbeddings(f.store, { force: true, collection: "docs" })).rejects.toThrow(/legacy|force|space|collection/i);
    expect(f.store.db.prepare("SELECT * FROM content_vectors").all()).toEqual(before);
  } finally { f.store.close(); }
});
test("a non-forced empty legacy store does not prepare an unused model", async () => {
  const store = createStore(":memory:"), llm = new FixtureGguf(2); store.llm = llm;
  try {
    expect((await generateEmbeddings(store)).docsProcessed).toBe(0);
    expect(llm.calls).toBe(0);
  } finally { store.close(); }
});
