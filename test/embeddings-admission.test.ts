import { expect, test } from "vitest";
import { planEmbeddingAdmission, type StoredEmbeddingSpace } from "../src/embeddings/space-admission.js";
import { embeddingSpaceId } from "../src/embeddings/identity.js";
import { makeSpaceDescriptor } from "./_helpers/embedding-config.js";
import type { EmbeddingIdentity } from "../src/embeddings/types.js";

function onnx(prefix = ""): Extract<EmbeddingIdentity, { kind: "onnx-v1" }> {
  const base = makeSpaceDescriptor(), descriptor = { ...base, recipe: { ...base.recipe, queryPrefix: prefix } };
  return { kind: "onnx-v1", id: embeddingSpaceId(descriptor), descriptor };
}
const legacy: Extract<EmbeddingIdentity, { kind: "legacy-gguf" }> = { kind: "legacy-gguf", model: "old.gguf", fingerprint: "abc123", dimensions: 2 };
test("empty vector reads do not initialize metadata; first embedding may initialize", () => {
  const producer = onnx();
  expect(planEmbeddingAdmission({ kind: "empty" }, producer, { kind: "query" }).action).toBe("read");
  expect(planEmbeddingAdmission({ kind: "empty" }, producer, { kind: "embed" }).action).toBe("initialize");
});
test("compatible ONNX producer permits read/write but retains an immutable expected state token", () => {
  const producer = onnx(), current: StoredEmbeddingSpace = { kind: "onnx-v1", identity: producer };
  const read = planEmbeddingAdmission(current, producer, { kind: "query" });
  const write = planEmbeddingAdmission(current, producer, { kind: "embed", force: true });
  expect(read.action).toBe("read"); expect(write.action).toBe("compatible-write");
  expect(read.expected).toBe(write.expected); expect(Object.isFrozen(write)).toBe(true);
});
test.each([{ force: false }, { force: true }, { force: true, collection: "docs" }])("same dimension cannot authorize a different ONNX recipe", operation => {
  expect(() => planEmbeddingAdmission({ kind: "onnx-v1", identity: onnx() }, onnx("different"), { kind: "embed", ...operation }))
    .toThrow(/separate database|space/i);
});
test.each([{ dimensions: 2 }, { dimensions: 3 }])("identified legacy unscoped full force preserves model/dimension replacement", ({ dimensions }) => {
  const producer: EmbeddingIdentity = { ...legacy, model: "new.gguf", fingerprint: "def456", dimensions };
  const result = planEmbeddingAdmission(legacy, producer, { kind: "embed", force: true });
  expect(result.action).toBe("legacy-full-force"); expect(result.identity).toBe(producer);
  expect(() => planEmbeddingAdmission(legacy, producer, { kind: "embed", force: true, collection: "docs" })).toThrow();
  expect(() => planEmbeddingAdmission(legacy, producer, { kind: "query", force: true })).toThrow();
});
test("compatible legacy operations retain old fingerprint admission rather than adopting a descriptor", () => {
  expect(planEmbeddingAdmission(legacy, legacy, { kind: "query" }).action).toBe("read");
  expect(planEmbeddingAdmission(legacy, legacy, { kind: "embed" }).action).toBe("compatible-write");
});
test("unknown/mixed state cannot use the legacy full-force exception", () => {
  expect(() => planEmbeddingAdmission({ kind: "ambiguous", reason: "mixed legacy rows" }, legacy, { kind: "embed", force: true }))
    .toThrow(/ambiguous|mixed|identity/i);
});
test("cross-backend transitions require a separate database even with full force", () => {
  expect(() => planEmbeddingAdmission(legacy, onnx(), { kind: "embed", force: true })).toThrow(/separate database|space/i);
  expect(() => planEmbeddingAdmission({ kind: "onnx-v1", identity: onnx() }, legacy, { kind: "embed", force: true })).toThrow(/separate database|space/i);
});
test("a mislabeled descriptor hash is rejected before planning", () => {
  const producer = onnx();
  expect(() => planEmbeddingAdmission({ kind: "empty" }, { ...producer, id: "onnx-v1:" + "0".repeat(64) }, { kind: "embed" }))
    .toThrow(/identity|hash|descriptor/i);
});
