import { describe, test, expect } from "vitest";
import { createEmbeddingStoreFixture } from "./_helpers/embedding-store.js";
import { createStore, hybridQuery, structuredSearch, vectorSearchQuery, generateEmbeddings, upsertStoreCollection, insertContent, insertDocument } from "../src/store.js";
import type { SearchHooks } from "../src/store.js";
import { controlledIndexingFixture } from "./_helpers/indexing-store.js";

describe("ONNX CPU policy", () => {
  test("unconfigured legacy search retains independently enabled expansion and reranking defaults", async () => {
    const store = createStore(":memory:");
    let expansions = 0, reranks = 0;
    try {
      upsertStoreCollection(store.db, "docs", { path: "/fixture/docs" });
      insertContent(store.db, "legacy-query-fixture", "alpha beta", "2026-01-01");
      insertDocument(store.db, "docs", "alpha.md", "Alpha", "legacy-query-fixture", "2026-01-01", "2026-01-01");
      store.expandQuery = async () => { expansions++; return []; };
      store.rerank = async (_query, documents) => { reranks++; return documents.map(doc => ({ file: doc.file, score: 1 })); };
      expect((await hybridQuery(store, "alpha", { intent: "avoid strong-signal bypass" })).map(result => result.file)).toEqual(["qmd://docs/alpha.md"]);
      expect(expansions).toBe(1); expect(reranks).toBe(1);
    } finally { store.close(); }
  });
  test("configured acceleration keeps independent stages even when initialization provenance is CPU retry", async () => {
    const f = controlledIndexingFixture(undefined, "cuda");
    let expansions = 0, reranks = 0;
    try {
      await generateEmbeddings(f.store);
      f.store.expandQuery = async () => { expansions++; return []; };
      f.store.rerank = async (_query, documents) => { reranks++; return documents.map(doc => ({ file: doc.file, score: 1 })); };
      expect(f.store.embedding?.provenance?.initializedMode).toBe("cpu");
      await hybridQuery(f.store, "alpha", { intent: "avoid strong-signal bypass" });
      expect(expansions).toBe(1); expect(reranks).toBe(1);
      expect((await vectorSearchQuery(f.store, "alpha")).map(result => result.file)).toEqual(["qmd://docs/alpha.md"]);
      expect(expansions).toBe(2); expect(reranks).toBe(1);
      await hybridQuery(f.store, "alpha", { expand: false, skipRerank: true });
      expect(expansions).toBe(2); expect(reranks).toBe(1);
      expect(f.events.some(event => event.startsWith("gguf-"))).toBe(false);
    } finally { await f.close(); }
  });
  test("1 default CPU hybrid: embedded query, no forbidden stages, result present", async () => {
    const f = createEmbeddingStoreFixture();
    try {
      await generateEmbeddings(f.store);
      f.inputs.length = 0;
      f.events.length = 0;

      const hooks: SearchHooks = {
        onExpandStart: () => { throw new Error("forbidden-expand-start"); },
        onRerankStart: () => { throw new Error("forbidden-rerank-start"); },
      };
      const res = await hybridQuery(f.store, "alpha", { hooks });
      expect(res.some(r => r.file === "qmd://docs/alpha.md")).toBe(true);
      expect(f.inputs).toEqual([{ kind: "query", text: "alpha" }]);
      expect(f.events.some(e => e.startsWith("gguf-"))).toBe(false);
    } finally {
      await f.close();
    }
  });

  test("2 explicit rerank enabling under CPU rejects policy before gguf/embedding", async () => {
    const f = createEmbeddingStoreFixture();
    try {
      f.inputs.length = 0;
      f.events.length = 0;
      const hybridFailure: unknown = await hybridQuery(f.store, "alpha", { skipRerank: false }).catch((error: unknown) => error);
      expect(f.inputs.length).toBe(0);
      expect(f.events.length).toBe(0);
      expect(String(hybridFailure)).toMatch(/CPU.*policy/i);

      f.inputs.length = 0;
      f.events.length = 0;
      const structuredFailure: unknown = await structuredSearch(f.store, [{ type: "vec", query: "alpha" }], { skipRerank: false }).catch((error: unknown) => error);
      expect(f.inputs.length).toBe(0);
      expect(f.events.length).toBe(0);
      expect(String(structuredFailure)).toMatch(/CPU.*policy/i);
    } finally {
      await f.close();
    }
  });

  test("3 explicit expand:true under CPU hybrid must reject before gguf/embedding", async () => {
    const f = createEmbeddingStoreFixture();
    try {
      f.inputs.length = 0;
      f.events.length = 0;
      const failure: unknown = await hybridQuery(f.store, "alpha", { expand: true }).catch((error: unknown) => error);
      expect(f.inputs.length).toBe(0);
      expect(f.events.length).toBe(0);
      expect(String(failure)).toMatch(/CPU.*policy/i);
    } finally {
      await f.close();
    }
  });

  test("4 manual expandQuery and rerank reject CPU policy and leave no gguf events", async () => {
    const f = createEmbeddingStoreFixture();
    try {
      const expansionFailure: unknown = await f.store.expandQuery("alpha").catch((error: unknown) => error);
      expect(f.events.some(e => e.startsWith("gguf-"))).toBe(false);
      expect(String(expansionFailure)).toMatch(/CPU.*policy/i);

      f.events.length = 0;
      const rerankFailure: unknown = await f.store.rerank("alpha", [{ file: "qmd://docs/alpha.md", text: "alpha" }]).catch((error: unknown) => error);
      expect(f.events.some(e => e.startsWith("gguf-"))).toBe(false);
      expect(String(rerankFailure)).toMatch(/CPU.*policy/i);
    } finally {
      await f.close();
    }
  });
});
