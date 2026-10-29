import { createHash } from "node:crypto";
import type { EmbeddingSpaceDescriptor, OnnxEmbeddingConfig } from "../../src/embeddings/types.js";

export function makeOnnxConfig(overrides: Partial<OnnxEmbeddingConfig> = {}): OnnxEmbeddingConfig {
  return {
    backend: "onnx",
    source: { kind: "huggingface", model: "fixture/model", revision: "a".repeat(40) },
    recipe: { version: "qmd-embedding-v1", pooling: "mean", normalization: "l2",
      queryPrefix: "", documentPrefix: "", maxInputTokens: 256, titleBudgetTokens: 32, overlapTokens: 24 },
    ...overrides,
  };
}

export function makeSpaceDescriptor(): EmbeddingSpaceDescriptor {
  const config = makeOnnxConfig();
  const files = ["config.json", "onnx/model.onnx", "tokenizer.json"].map(path => ({
    bytes: Buffer.byteLength(path), path, sha256: createHash("sha256").update(path).digest("hex"),
  }));
  // This fixed-order fixture is independent of the production canonical serializer.
  const digest = createHash("sha256").update(JSON.stringify({ files, source: config.source, version: 1 })).digest("hex");
  return {
    version: 1, backend: "onnx", artifactManifest: { version: 1, source: config.source, files, digest },
    dtype: "fp32", nativeDimensions: 2, dimensions: 2, recipe: config.recipe,
    formattingVersion: "qmd-onnx-format-v1", tokenizerBehaviorVersion: "qmd-onnx-tokenizer-v1",
    chunking: { version: "qmd-onnx-chunk-v1", maxInputTokens: 256, titleBudgetTokens: 32, overlapTokens: 24 },
  };
}
