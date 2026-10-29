import { DEFAULT_MODEL_CACHE_DIR } from "../llm.js";
import type { LlamaCpp } from "../llm.js";
import { join } from "node:path";
import type { ArtifactIO } from "./artifacts.js";
import { createArtifactIO } from "./artifact-io.js";
import { createGgufEmbeddingCapability, type LegacyEmbeddingFormatting } from "./gguf.js";
import { leaseEmbeddingCapability } from "./lifecycle.js";
import { createOnnxEmbeddingCapability } from "./onnx.js";
import { createTransformersRuntime } from "./onnx-transformers.js";
import type { OnnxRuntimePort } from "./onnx-runtime-types.js";
import type { EmbeddingCapability, ResolvedEmbeddingConfig } from "./types.js";

function assertNever(value: never): never { throw new Error(`Unsupported embedding backend: ${String(value)}`); }

export function createEmbeddingCapability(
  config: ResolvedEmbeddingConfig, llm: LlamaCpp, formatting: LegacyEmbeddingFormatting,
  io?: ArtifactIO, runtime?: OnnxRuntimePort,
): EmbeddingCapability {
  switch (config.backend) {
    case "gguf":
      return leaseEmbeddingCapability(createGgufEmbeddingCapability(llm, config.model, formatting));
    case "onnx": {
      const artifactIO = io ?? createArtifactIO(join(DEFAULT_MODEL_CACHE_DIR, "onnx"));
      const onnxRuntime = runtime ?? createTransformersRuntime();
      return leaseEmbeddingCapability(createOnnxEmbeddingCapability(config, artifactIO, onnxRuntime));
    }
    default: return assertNever(config);
  }
}
