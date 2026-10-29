import { expect, test } from "vitest";
import { generateEmbeddings } from "../src/store.js";
import { controlledIndexingFixture } from "./_helpers/indexing-store.js";
import { deferred } from "./_helpers/deferred.js";

test("a failed native batch retries raw chunks with the same producer and commits ordered identities", async () => {
  const f = controlledIndexingFixture(async inputs => { if (inputs.length > 1) throw new Error("fixture batch failure"); });
  try {
    const result = await generateEmbeddings(f.store);
    expect(result.chunksEmbedded).toBe(2); expect(result.errors).toBe(0);
    expect(f.calls.map(call => call.length)).toEqual([2, 1, 1]);
    expect(f.calls.flat().every(input => input.kind === "document")).toBe(true);
    expect(f.store.db.prepare("SELECT seq,total_chunks FROM content_vectors ORDER BY seq").all()).toEqual([{ seq: 0, total_chunks: 2 }, { seq: 1, total_chunks: 2 }]);
    expect(f.events.some(event => event.startsWith("gguf-"))).toBe(false);
  } finally { await f.close(); }
});
test("inflight cancellation withholds all batch results and writes no vector/space metadata", async () => {
  const started = deferred<void>(), release = deferred<void>(), controller = new AbortController();
  const f = controlledIndexingFixture(async () => { started.resolve(); await release.promise; });
  const pending = generateEmbeddings(f.store, { signal: controller.signal }).catch((error: unknown) => error);
  try {
    const phase = await Promise.race([started.promise.then(() => "started"), pending.then(() => "settled")]);
    expect(phase).toBe("started"); controller.abort("fixture index cancellation"); release.resolve();
    expect(await pending).toBe("fixture index cancellation");
    expect(f.store.db.prepare("SELECT * FROM content_vectors").all()).toEqual([]);
    expect(f.store.db.prepare("SELECT key FROM store_config WHERE key='embedding_space_v1'").all()).toEqual([]);
    expect(f.calls).toHaveLength(1);
  } finally { release.resolve(); await pending; await f.close(); }
});
test("database insertion failure rolls back the batch and does not become individual retry success", async () => {
  const f = controlledIndexingFixture();
  try {
    f.store.db.exec("CREATE TRIGGER fixture_insert_failure BEFORE INSERT ON content_vectors WHEN NEW.seq=1 BEGIN SELECT RAISE(ABORT,'fixture insert failure'); END");
    await expect(generateEmbeddings(f.store)).rejects.toThrow(/fixture insert failure|persist|transaction/i);
    expect(f.store.db.prepare("SELECT * FROM content_vectors").all()).toEqual([]);
    expect(f.store.db.prepare("SELECT * FROM vector_rows").all()).toEqual([]);
    expect(f.store.db.prepare("SELECT key FROM store_config WHERE key='embedding_space_v1'").all()).toEqual([]);
    expect(f.calls).toHaveLength(1);
  } finally { await f.close(); }
});
test("permanent chunk failure removes an incomplete hash and reports only outstanding retry errors", async () => {
  const f = controlledIndexingFixture(async inputs => {
    if (inputs.length > 1 || inputs.some(input => input.text === "beta")) throw new Error("fixture permanent chunk failure");
  });
  try {
    const result = await generateEmbeddings(f.store);
    expect(result.errors).toBe(1); expect(result.failures?.[0]?.attempts).toBe(3);
    expect(result.chunksEmbedded).toBe(0);
    expect(f.store.db.prepare("SELECT * FROM content_vectors").all()).toEqual([]);
    expect(f.store.db.prepare("SELECT * FROM vector_rows").all()).toEqual([]);
  } finally { await f.close(); }
});
test("store close drains the complete indexing operation through its final database writes", async () => {
  const started = deferred<void>(), release = deferred<void>();
  const f = controlledIndexingFixture(async () => { started.resolve(); await release.promise; });
  const indexing = generateEmbeddings(f.store);
  let closing: Promise<void> | undefined, closed = false;
  try {
    const phase = await Promise.race([started.promise.then(() => "started"), indexing.then(() => "settled", () => "failed")]);
    expect(phase).toBe("started");
    closing = f.store.closeAsync().then(() => { closed = true; });
    await Promise.resolve(); expect(closed).toBe(false);
    release.resolve(); expect((await indexing).chunksEmbedded).toBe(2);
    await closing; expect(closed).toBe(true);
    expect(() => f.store.db.prepare("SELECT 1").get()).toThrow();
  } finally { release.resolve(); await Promise.allSettled([indexing, ...(closing ? [closing] : [])]); await f.close(); }
});
