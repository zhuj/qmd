import { expect, test } from "vitest";
import { createArtifactManifest, embeddingSpaceId, parseEmbeddingSpaceDescriptor } from "../src/embeddings/identity.js";
import { makeSpaceDescriptor } from "./_helpers/embedding-config.js";

test("descriptor parser returns typed frozen provenance without changing canonical identity", () => {
  const base = makeSpaceDescriptor();
  const source = { kind: "local", path: "/fixture/complete-model" } satisfies typeof base.artifactManifest.source;
  const input = { ...base, artifactManifest: createArtifactManifest(source, base.artifactManifest.files) };
  const json = JSON.stringify(input), raw: unknown = JSON.parse(json);
  const parsed = parseEmbeddingSpaceDescriptor(raw);
  expect(parsed.artifactManifest.source).toEqual(source);
  expect(embeddingSpaceId(parsed)).toBe(embeddingSpaceId(input));
  expect(JSON.stringify(raw)).toBe(json);
  expect(Object.isFrozen(parsed)).toBe(true); expect(Object.isFrozen(parsed.recipe)).toBe(true);
  expect(Object.isFrozen(parsed.chunking)).toBe(true); expect(Object.isFrozen(parsed.artifactManifest)).toBe(true);
});
test.each([
  { change: { version: 2 } }, { change: { dimensions: 3 } }, { change: { backend: "gguf" } },
  { change: { extra: "unsupported" } }, { change: { recipe: { version: "wrong" } } },
])("stored descriptor boundary rejects malformed or unsupported data", ({ change }) => {
  expect(() => parseEmbeddingSpaceDescriptor({ ...makeSpaceDescriptor(), ...change })).toThrow();
});
