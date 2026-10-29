import { isBun } from "../../src/db.js";
import { createStore } from "../../src/store.js";
import { checkEmbeddingVectorSamples } from "../../src/cli/qmd.js";
import { LlamaCpp, setDefaultLlamaCpp } from "../../src/llm.js";
import { verifyHeapLimit } from "./heap-limit.js";

// Every stored and reproduced vector is [1, 0], isolating sampling and chunking.
class ConstantLlm extends LlamaCpp {
  override async tokenize(text: string) { return new Array(Math.ceil(text.length / 16)).fill(1); }
  override async embed() { return { embedding: [1, 0], model: "model" }; }
}

const strict = process.env.QMD_TEST_SQLITE_MEMORY_STRICT === "1";
const budgetBytes = 33554432;

const store = createStore(":memory:");
try {
  setDefaultLlamaCpp(new ConstantLlm());
  const body = "word ".repeat(13_107); // About 64 KiB per document.
  store.ensureVecTable(2);
  store.db.transaction(() => {
    for (let doc = 0; doc < 16; doc++) {
      const hash = `document-${doc}`;
      store.insertContent(hash, body, "2026-01-01");
      for (let path = 0; path < 8; path++) {
        store.insertDocument("test", `${hash}-${path}.md`, hash, hash, "2026-01-01", "2026-01-01");
      }
      for (let seq = 0; seq < 32; seq++) {
        store.insertEmbedding(hash, seq, 0, new Float32Array([1, 0]), "model", "2026-01-01", 32, "current");
      }
    }
  })();
  store.db.exec("PRAGMA temp_store = MEMORY");
  store.db.exec(`PRAGMA hard_heap_limit = ${budgetBytes}`);

  const binds = verifyHeapLimit(store.db, budgetBytes, strict);

  const result = await checkEmbeddingVectorSamples(store.db, "model", "current");
  console.log(JSON.stringify({ ...result, limitBinds: binds, runtime: isBun ? "bun" : "node", budgetBytes }));
} finally {
  setDefaultLlamaCpp(null);
  store.close();
}
