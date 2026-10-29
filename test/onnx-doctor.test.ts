import { describe, test, expect } from "vitest";
import { createEmbeddingStoreFixture } from "./_helpers/embedding-store.js";
import { createStore, maybeAdoptLegacyEmbeddingFingerprint, EmbeddingCompatibilityUnverifiedError, getIndexHealth, inspectStoreEmbeddings } from "../src/store.js";
import { planEmbeddingAdmission, readEmbeddingSpace, commitEmbeddingSpace } from "../src/embeddings/space-store.js";
import { resolveEmbeddingConfig } from "../src/embeddings/config.js";
import { embeddingSpaceId } from "../src/embeddings/identity.js";
import type { EmbeddingCapability, EmbeddingIdentity, EmbeddingInput, RuntimeProvenance } from "../src/embeddings/types.js";
import { makeOnnxConfig, makeSpaceDescriptor } from "./_helpers/embedding-config.js";
import { deferred } from "./_helpers/deferred.js";

describe("onnx negative adoption guard (T16a)", () => {
  test("configured ONNX with no vectors -> exact negative guard", async () => {
    const fixture = createEmbeddingStoreFixture();
    try {
      expect(fixture.events.length).toBe(0);
      const result = await maybeAdoptLegacyEmbeddingFingerprint(fixture.store, "fixture.gguf");
      expect(result.checked).toBe(false);
      expect(result.adopted).toBe(0);
      expect(result.reason).toMatch(/ONNX.*cannot adopt.*separate database/);
      expect(fixture.events).toEqual([]);
    } finally {
      await fixture.close();
    }
  });

  test("configured ONNX with legacy empty-fp rows -> guard blocks before ForbiddenLlama", async () => {
    const fixture = createEmbeddingStoreFixture();
    try {
      const db = fixture.store.db;
      fixture.store.ensureVecTable(2);
      fixture.store.insertEmbedding("fixture-alpha-beta", 0, 0, new Float32Array([1, 0]), "fixture.gguf", "2026-01-01", 1, "");
      const beforeContent = db.prepare("SELECT * FROM content_vectors ORDER BY rowid").all();
      const beforeVec = db.prepare("SELECT * FROM vector_rows ORDER BY rowid").all();
      const beforeConfig = db.prepare("SELECT * FROM store_config ORDER BY key").all();
      expect(fixture.events.length).toBe(0);

      const result = await maybeAdoptLegacyEmbeddingFingerprint(fixture.store, "fixture.gguf");

      const afterContent = db.prepare("SELECT * FROM content_vectors ORDER BY rowid").all();
      const afterVec = db.prepare("SELECT * FROM vector_rows ORDER BY rowid").all();
      const afterConfig = db.prepare("SELECT * FROM store_config ORDER BY key").all();
      expect(afterContent).toEqual(beforeContent);
      expect(afterVec).toEqual(beforeVec);
      expect(afterConfig).toEqual(beforeConfig);
      expect(fixture.events).toEqual([]);
      expect(result.checked).toBe(false);
      expect(result.adopted).toBe(0);
      expect(result.reason).toMatch(/ONNX.*cannot adopt.*separate database/);
    } finally {
      await fixture.close();
    }
  });

  test("bare legacy store -> legacy branch untouched (no ONNX guard needed)", async () => {
    const store = createStore(":memory:");
    try {
      const result = await maybeAdoptLegacyEmbeddingFingerprint(store, "fixture.gguf");
      expect(result.checked).toBe(false);
      expect(result.adopted).toBe(0);
      expect(result.reason).toBe("no legacy empty-fingerprint embeddings");
    } finally {
      store.close();
    }
  });
});

describe("onnx health identity guard (T16b1)", () => {
  test("A configured fixture UNPREPARED -> throws EmbeddingCompatibilityUnverifiedError", async () => {
    const fixture = createEmbeddingStoreFixture();
    try {
      expect(fixture.events.length).toBe(0);
      expect(fixture.store.db.prepare("SELECT * FROM content_vectors").all()).toEqual([]);
      expect(() => fixture.store.getIndexHealth()).toThrow(EmbeddingCompatibilityUnverifiedError);
      expect(() => fixture.store.getHashesNeedingEmbedding()).toThrow(EmbeddingCompatibilityUnverifiedError);
      expect(fixture.events).toEqual([]);
    } finally {
      await fixture.close();
    }
  });

  test("B fixture prepare -> identity cached, health allowed with explicit full id", async () => {
    const fixture = createEmbeddingStoreFixture();
    try {
      if (!fixture.store.embedding) throw new Error("fixture embedding missing");
      const identity = await fixture.store.embedding.prepare();
      if (identity.kind !== "onnx-v1") throw new Error("Expected ONNX fixture identity");
      expect(fixture.events).toEqual(["prepare"]);
      const health = fixture.store.getIndexHealth("wrong-display");
      expect(health.needsEmbedding).toBe(1);
      expect(health.totalDocs).toBe(1);
      expect(typeof health.daysStale).toBe("number");
      const hashes = fixture.store.getHashesNeedingEmbedding();
      expect(hashes).toBe(1);
      fixture.store.db.transaction(() => {
        commitEmbeddingSpace(fixture.store.db, planEmbeddingAdmission(readEmbeddingSpace(fixture.store.db), identity, { kind: "embed" }));
        fixture.store.ensureVecTable(2);
        fixture.store.insertEmbedding("fixture-alpha-beta", 0, 0, new Float32Array([0.6, 0.8]), "different-display", "2026-01-01", 1, identity.id);
      }).immediate();
      expect(fixture.store.getIndexHealth("wrong-display").needsEmbedding).toBe(0);
      expect(fixture.store.getHashesNeedingEmbedding("wrong-display")).toBe(0);
      expect(fixture.events).toEqual(["prepare"]);
    } finally {
      await fixture.close();
    }
  });

  test("C raw DB with persisted ONNX descriptor and no current producer proof -> unverified even with coherent namespace", async () => {
    const fixture = createEmbeddingStoreFixture();
    try {
      if (!fixture.store.embedding) throw new Error("fixture embedding missing");
      const identity = await fixture.store.embedding.prepare();
      const db = fixture.store.db;
      db.transaction(() => commitEmbeddingSpace(db, planEmbeddingAdmission(readEmbeddingSpace(db), identity, { kind: "embed" }))).immediate();
      const space = readEmbeddingSpace(db);
      expect(space.kind).toBe("onnx-v1");
      // Free raw DB call without identity/proof -> unverified
      expect(() => getIndexHealth(db, "display-label")).toThrow(EmbeddingCompatibilityUnverifiedError);
      const health = getIndexHealth(db, "display-label", identity);
      expect(health.needsEmbedding).toBe(1);
    } finally {
      await fixture.close();
    }
  });
});

function diagnosticFixture() {
  const config = resolveEmbeddingConfig(makeOnnxConfig({ device: "cuda" }), { defaultGgufModel: "unused", hostParallelism: 8 });
  const descriptor = makeSpaceDescriptor();
  const identity: Extract<EmbeddingIdentity, { kind: "onnx-v1" }> = { kind: "onnx-v1", id: embeddingSpaceId(descriptor), descriptor };
  const trace: string[] = [], inputs: EmbeddingInput[] = [];
  const state: { live?: RuntimeProvenance; vectors: readonly Float32Array[]; wait?: Promise<void>; entered?: () => void; preparationError?: Error } = {
    vectors: [new Float32Array([0.6, 0.8])],
  };
  const raw: EmbeddingCapability = {
    config, modelLabel: "diagnostic-model", get provenance() { return state.live; },
    prepare: async () => { trace.push("prepare"); if (state.preparationError) throw state.preparationError; return identity; },
    encode: async text => { trace.push("encode"); return Array.from(text, c => c.codePointAt(0) ?? 0); },
    decode: async tokens => { trace.push("decode"); return String.fromCodePoint(...tokens); },
    measureTokens: async input => {
      trace.push("measure"); return { text: input.text.length, total: input.text.length + 2 + ("title" in input ? input.title.length : 0) };
    },
    embed: async values => {
      trace.push("embed"); inputs.push(...values); state.entered?.(); if (state.wait) await state.wait; return state.vectors;
    },
    close: async () => { trace.push("close"); },
  };
  return { ...createEmbeddingStoreFixture(config, raw), identity, trace, inputs, state };
}
function seedDiagnostic(f: ReturnType<typeof diagnosticFixture>, identity: EmbeddingIdentity = f.identity): void {
  f.store.db.transaction(() => {
    commitEmbeddingSpace(f.store.db, planEmbeddingAdmission(readEmbeddingSpace(f.store.db), identity, { kind: "embed" }), {
      requestedDevice: "cuda", initializedMode: "cuda", graphNodePlacement: "unknown",
    });
    f.store.ensureVecTable(2);
    f.store.insertEmbedding("fixture-alpha-beta", 7, 0, new Float32Array([0.6, 0.8]), "another-display", "2026-01-01", 1,
      identity.kind === "onnx-v1" ? identity.id : identity.fingerprint);
  }).immediate();
}

describe("T16 selected-producer diagnostics", () => {
  test("failed selected preparation reports unverified artifact guidance without leaking its cause", async () => {
    const f = diagnosticFixture();
    try {
      f.state.preparationError = new Error("missing /sensitive/model and private document text");
      const result = await inspectStoreEmbeddings(f.store, true);
      expect(result).toMatchObject({ compatible: false, sampleVerification: "failed" });
      expect(result.reason).toMatch(/unverified.*artifacts/i);
      expect(JSON.stringify(result)).not.toMatch(/sensitive|private document/);
      expect(f.trace).toEqual(["prepare"]); expect(f.events).toEqual([]);
    } finally { await f.close(); }
  });
  test("metadata inspection inventories stored identity but does not infer live initialization or mutate SQL", async () => {
    const f = diagnosticFixture();
    try {
      seedDiagnostic(f);
      const before = f.store.db.prepare("SELECT * FROM store_config ORDER BY key").all();
      const rows = f.store.db.prepare("SELECT * FROM content_vectors").all();
      const mappings = f.store.db.prepare("SELECT * FROM vector_rows ORDER BY id").all();
      const result = await inspectStoreEmbeddings(f.store);
      expect(result).toMatchObject({ backend: "onnx", identity: undefined, storedIdentity: f.identity.id,
        compatible: false, requestedDevice: "cuda", initializedMode: undefined, graphNodePlacement: "unknown", sampleVerification: "not-run" });
      expect(result.reason).toMatch(/unverified|not established/i);
      f.state.live = { requestedDevice: "cpu", initializedMode: "cpu", cpuRetryReason: "provider-initialization-failed", graphNodePlacement: "unknown" };
      expect(await inspectStoreEmbeddings(f.store)).toMatchObject({ compatible: false, requestedDevice: "cuda", initializedMode: "cpu", cpuRetryReason: "provider-initialization-failed" });
      expect(f.trace).toEqual([]); expect(f.events).toEqual([]);
      expect(f.store.db.prepare("SELECT * FROM store_config ORDER BY key").all()).toEqual(before);
      expect(f.store.db.prepare("SELECT * FROM content_vectors").all()).toEqual(rows);
      expect(f.store.db.prepare("SELECT * FROM vector_rows ORDER BY id").all()).toEqual(mappings);
    } finally { await f.close(); }
  });
  test.each([false, true])("cached producer admits only the same semantic namespace; changed recipe=%s", async changed => {
    const f = diagnosticFixture();
    try {
      if (!f.store.embedding) throw new Error("fixture producer missing");
      await f.store.embedding.prepare();
      const descriptor = { ...f.identity.descriptor, recipe: { ...f.identity.descriptor.recipe, queryPrefix: changed ? "changed:" : "" } };
      seedDiagnostic(f, { kind: "onnx-v1", id: embeddingSpaceId(descriptor), descriptor });
      const result = await inspectStoreEmbeddings(f.store);
      expect(result.compatible).toBe(!changed);
      if (changed) expect(result.reason).toMatch(/separate database/i);
      expect(f.trace).toEqual(["prepare"]);
      if (changed) { expect((await inspectStoreEmbeddings(f.store, true)).compatible).toBe(false); expect(f.inputs).toEqual([]); }
      expect(f.events).toEqual([]);
    } finally { await f.close(); }
  });
  const samples = [
    { label: "exact", vectors: [new Float32Array([0.6, 0.8])], status: "exact-reproduction-observed" },
    { label: "different", vectors: [new Float32Array([0.8, 0.6])], status: "exact-reproduction-not-observed" },
    { label: "missing", vectors: [], status: "failed" },
    { label: "dimension", vectors: [new Float32Array([1, 0, 0])], status: "failed" },
    { label: "nonfinite", vectors: [new Float32Array([NaN, 0])], status: "failed" },
    { label: "zero", vectors: [new Float32Array([0, 0])], status: "failed" },
  ];
  test.each(samples)("sample $label uses saved position, raw document formatting, and validates output independently of identity", async ({ vectors, status }) => {
    const f = diagnosticFixture();
    try {
      seedDiagnostic(f); f.state.vectors = vectors;
      const result = await inspectStoreEmbeddings(f.store, true);
      expect(result.compatible).toBe(true);
      expect(result.sampleVerification).toBe(status);
      expect(f.inputs).toEqual([{ kind: "document", text: "alpha beta", title: "alpha" }]);
      expect(f.trace).toContain("measure"); expect(f.events).toEqual([]);
      expect(JSON.stringify(result)).not.toMatch(/alpha beta|fixture\/model|force/i);
    } finally { await f.close(); }
  });
  test.each([false, true])("diagnostic lease holds through final namespace reads; namespace changes=%s", async changed => {
    const f = diagnosticFixture(), entered = deferred<void>(), release = deferred<void>();
    f.state.entered = () => entered.resolve(); f.state.wait = release.promise;
    try {
      seedDiagnostic(f);
      const inspection = inspectStoreEmbeddings(f.store, true);
      await Promise.race([entered.promise, inspection.then(() => { throw new Error("diagnostic finished before embedding"); })]);
      const closed = f.close();
      expect(f.trace).not.toContain("close");
      if (changed) f.store.db.prepare("UPDATE store_config SET value='{}' WHERE key='embedding_space_v1'").run();
      release.resolve();
      const result = await inspection;
      expect(result.compatible).toBe(!changed);
      if (changed) expect(result.reason).toMatch(/separate database/i);
      await closed;
      expect(f.trace.at(-1)).toBe("close");
    } finally { release.resolve(); await f.close(); }
  });
  test.each([false, true])("unbound persisted new metadata cannot use legacy adoption; malformed=%s", async malformed => {
    const store = createStore(":memory:"), descriptor = makeSpaceDescriptor();
    try {
      if (malformed) store.db.prepare("INSERT INTO store_config(key,value) VALUES('embedding_space_v1','{}')").run();
      else store.db.transaction(() => commitEmbeddingSpace(store.db,
        planEmbeddingAdmission(readEmbeddingSpace(store.db), { kind: "onnx-v1", id: embeddingSpaceId(descriptor), descriptor }, { kind: "embed" }))).immediate();
      expect((await maybeAdoptLegacyEmbeddingFingerprint(store, "old.gguf")).reason).toMatch(/separate database/i);
      expect(() => getIndexHealth(store.db, "old.gguf")).toThrow(EmbeddingCompatibilityUnverifiedError);
      expect(await inspectStoreEmbeddings(store)).toMatchObject({ backend: "onnx", compatible: false, sampleVerification: "not-run" });
    } finally { await store.closeAsync(); }
  });
});
