import { createHash } from "node:crypto";
import { expect, test } from "vitest";
import { createArtifactManifest, embeddingSpaceId, serializeEmbeddingSpace } from "../src/embeddings/identity.js";
import { makeSpaceDescriptor } from "./_helpers/embedding-config.js";

test("semantic IDs use the full SHA-256 of canonical UTF-8 descriptors", () => {
  const space = makeSpaceDescriptor();
  const serialized = serializeEmbeddingSpace(space);
  const digest = createHash("sha256").update(serialized, "utf8").digest("hex");
  expect(embeddingSpaceId(space)).toBe(`onnx-v1:${digest}`);
  expect(embeddingSpaceId(space)).toMatch(/^onnx-v1:[0-9a-f]{64}$/);
});

test("the v1 serialization preserves the complete fixed canonical contract", () => {
  const space = makeSpaceDescriptor();
  expect(serializeEmbeddingSpace(space)).toBe(JSON.stringify({
    artifactManifest: { digest: space.artifactManifest.digest, files: space.artifactManifest.files,
      source: { kind: "huggingface", model: "fixture/model", revision: "a".repeat(40) }, version: 1 },
    backend: "onnx", chunking: { maxInputTokens: 256, overlapTokens: 24, titleBudgetTokens: 32, version: "qmd-onnx-chunk-v1" },
    dimensions: 2, dtype: "fp32", formattingVersion: "qmd-onnx-format-v1", nativeDimensions: 2,
    recipe: { documentPrefix: "", maxInputTokens: 256, normalization: "l2", overlapTokens: 24,
      pooling: "mean", queryPrefix: "", titleBudgetTokens: 32, version: "qmd-embedding-v1" },
    tokenizerBehaviorVersion: "qmd-onnx-tokenizer-v1", version: 1,
  }));
});

test("canonical identity is independent of object and manifest entry insertion order", () => {
  const space = makeSpaceDescriptor();
  const { overlapTokens, ...otherRecipeFields } = space.recipe;
  const reordered = {
    ...space, recipe: { overlapTokens, ...otherRecipeFields },
    artifactManifest: { ...space.artifactManifest, files: [...space.artifactManifest.files].reverse() },
  };
  expect(serializeEmbeddingSpace(reordered)).toBe(serializeEmbeddingSpace(space));
});

test.each([
  { recipe: { queryPrefix: "query: " } }, { recipe: { documentPrefix: "passage: " } },
  { recipe: { queryPrefix: "检索: " } }, { recipe: { pooling: "last-token" } },
  { recipe: { pooling: "cls", clsTokenIndex: 0 } },
  { recipe: { pooling: "sentence-output", sentenceOutput: "sentence_embedding" } },
  { recipe: { normalization: "none" } }, { recipe: { maxInputTokens: 512 } },
  { recipe: { titleBudgetTokens: 48 } }, { dtype: "q8" },
  { chunking: { maxInputTokens: 192 } }, { chunking: { overlapTokens: 12 } },
  { chunking: { titleBudgetTokens: 16 } }, { dimensions: 3, nativeDimensions: 3 },
])("equal-sized semantic changes remain distinct: %j", change => {
  const space = makeSpaceDescriptor();
  const changed = { ...space, ...change, recipe: { ...space.recipe, ...("recipe" in change ? change.recipe : {}) },
    chunking: { ...space.chunking, ...("chunking" in change ? change.chunking : {}) } };
  expect(embeddingSpaceId(changed)).not.toBe(embeddingSpaceId(space));
});

test.each(["tokenizer.json", "onnx/model.onnx", "config.json"])(
  "changed %s bytes change identity even at identical dimensions", path => {
    const space = makeSpaceDescriptor();
    const manifest = createArtifactManifest(space.artifactManifest.source,
      space.artifactManifest.files.map(file => file.path === path ? { ...file, sha256: "b".repeat(64) } : file));
    expect(embeddingSpaceId({ ...space, artifactManifest: manifest })).not.toBe(embeddingSpaceId(space));
  },
);

test("identical local files retain identity after relocation while preserving source provenance", () => {
  const space = makeSpaceDescriptor();
  const first = createArtifactManifest({ kind: "local", path: "/models/a" }, space.artifactManifest.files);
  const second = createArtifactManifest({ kind: "local", path: "/elsewhere/b" }, space.artifactManifest.files);
  expect(first.source).toEqual({ kind: "local", path: "/models/a" });
  expect(second.source).toEqual({ kind: "local", path: "/elsewhere/b" });
  expect(first.digest).toBe(second.digest);
  expect(embeddingSpaceId({ ...space, artifactManifest: first }))
    .toBe(embeddingSpaceId({ ...space, artifactManifest: second }));
});

test.each([{ model: "fixture/model", revision: "b".repeat(40) }, { model: "fixture/other", revision: "a".repeat(40) }])(
  "HF repository and revision remain part of identity: %j", source => {
  const space = makeSpaceDescriptor();
  const manifest = createArtifactManifest({ kind: "huggingface", ...source },
    space.artifactManifest.files);
  expect(embeddingSpaceId({ ...space, artifactManifest: manifest })).not.toBe(embeddingSpaceId(space));
  },
);

test.each([
  { nativeDimensions: 0 }, { dimensions: -1 }, { dimensions: 1.5 }, { dimensions: Infinity },
  { dimensions: 1 }, { version: 2 }, { formattingVersion: "unknown" },
  { tokenizerBehaviorVersion: "unknown" }, { recipe: { version: "unknown" } },
  { chunking: { maxInputTokens: 300 } }, { chunking: { maxInputTokens: 40 } },
  { device: "cpu" }, { intraOpNumThreads: 2 },
])("invalid or operational descriptor fields are rejected: %j", change => {
  const space = makeSpaceDescriptor();
  const changed = { ...space, ...change, recipe: { ...space.recipe, ...("recipe" in change ? change.recipe : {}) },
    chunking: { ...space.chunking, ...("chunking" in change ? change.chunking : {}) } };
  expect(() => embeddingSpaceId(changed)).toThrow();
});

test("manifest digests are verified rather than trusted", () => {
  const space = makeSpaceDescriptor();
  expect(() => embeddingSpaceId({ ...space, artifactManifest: { ...space.artifactManifest, digest: "f".repeat(64) } }))
    .toThrow(/manifest.*digest/i);
});

test.each(["../weights.onnx", "/weights.onnx", "C:\\weights.onnx", "onnx/../weights.onnx", "bad\0path"])(
  "unsafe artifact paths cannot enter identity: %s", path => {
    expect(() => createArtifactManifest({ kind: "local", path: "/models" },
      [{ path, sha256: "a".repeat(64), bytes: 1 }])).toThrow(/path/i);
  },
);

test("normalized duplicate artifact paths are rejected", () => {
  expect(() => createArtifactManifest({ kind: "local", path: "/models" }, [
    { path: "onnx/model.onnx", sha256: "a".repeat(64), bytes: 1 },
    { path: "onnx\\model.onnx", sha256: "a".repeat(64), bytes: 1 },
  ])).toThrow(/duplicate/i);
});

test("manifest entries are normalized, sorted, validated and copied", () => {
  const entries = [{ path: "z\\weights.onnx", sha256: "A".repeat(64), bytes: 0 },
    { path: "config.json", sha256: "b".repeat(64), bytes: 1 }];
  const manifest = createArtifactManifest({ kind: "local", path: "/models" }, entries);
  entries.reverse();
  expect(manifest.files).toEqual([
    { path: "config.json", sha256: "b".repeat(64), bytes: 1 },
    { path: "z/weights.onnx", sha256: "a".repeat(64), bytes: 0 },
  ]);
  expect(Object.isFrozen(manifest.files)).toBe(true);
  expect(() => createArtifactManifest(manifest.source, [{ path: "x", sha256: "invalid", bytes: -1 }])).toThrow();
});
test("manifest construction parses an unknown JSON inventory at its boundary", () => {
  const space = makeSpaceDescriptor(), inventory: unknown = space.artifactManifest.files;
  expect(createArtifactManifest(space.artifactManifest.source, inventory)).toEqual(space.artifactManifest);
  const malformed: unknown = [{ path: "config.json", sha256: "bad", bytes: -1 }];
  expect(() => createArtifactManifest(space.artifactManifest.source, malformed)).toThrow();
});
