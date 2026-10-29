import { availableParallelism } from "node:os";
import { parseCollectionConfig, type CollectionConfig } from "../collections.js";
import { decideLocalConfigGate, type BuiltinModels, type ModelsSnapshot, type SensitiveSnapshot } from "../trust.js";
import type { ResolvedEmbeddingConfig } from "./types.js";
import { resolveEmbeddingConfig } from "./config.js";

export interface ApprovedEntryConfig {
  readonly configPath: string;
  readonly config: CollectionConfig;
  readonly embedding: ResolvedEmbeddingConfig;
  readonly digest: string;
}

function freezeCaptured(value: unknown, seen = new WeakSet<object>()): void {
  if (value === null || typeof value !== "object" || seen.has(value)) return;
  seen.add(value);
  for (const child of Object.values(value)) freezeCaptured(child, seen);
  Object.freeze(value);
}

export function approveEntryConfig(
  configPath: string, config: CollectionConfig, builtins: BuiltinModels,
  isInteractive: boolean, trustedCheck?: (configPath: string, digest: string) => boolean,
): ApprovedEntryConfig {
  const env = { ...process.env };
  const captured = parseCollectionConfig(structuredClone(config), configPath);
  const embedding = resolveEmbeddingConfig(captured.models?.embed, {
    configPath, defaultGgufModel: builtins.embed, legacyEnv: env.QMD_EMBED_MODEL,
    hostParallelism: availableParallelism(),
  });
  let trustEmbed: ModelsSnapshot["embed"];
  switch (embedding.backend) {
    case "onnx": trustEmbed = embedding; break;
    case "gguf": trustEmbed = typeof captured.models?.embed === "string" ? captured.models.embed : undefined; break;
    default: { const exhaustive: never = embedding; throw new Error(`Unsupported embedding configuration: ${exhaustive}`); }
  }
  const snapshot: SensitiveSnapshot = {
    hooks: Object.entries(captured.collections).flatMap(([collection, value]) => value.update ? [{ collection, command: value.update }] : []),
    paths: Object.entries(captured.collections).map(([collection, value]) => ({ collection, path: value.path })),
    models: { embed: trustEmbed, generate: captured.models?.generate, rerank: captured.models?.rerank },
  };
  const decision = decideLocalConfigGate({ configPath, snapshot, builtins, isInteractive, trustedCheck, env });
  if (decision.action !== "run") throw new Error("Project-local configuration approval required; review and trust this configuration before using its models.");
  switch (embedding.backend) {
    case "onnx": captured.models = { ...captured.models, embed: embedding }; break;
    case "gguf": captured.models = { ...captured.models, embed: embedding.model }; break;
    default: { const exhaustive: never = embedding; throw new Error(`Unsupported embedding configuration: ${exhaustive}`); }
  }
  freezeCaptured(captured);
  return Object.freeze({ configPath, config: captured, embedding, digest: decision.digest });
}
