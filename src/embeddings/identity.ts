import { createHash } from "node:crypto";
import { z } from "zod";
import { parseOnnxEmbeddingConfig } from "./config.js";
import type { ArtifactManifest, EmbeddingSpaceDescriptor, OnnxModelSource } from "./types.js";

function sha256(text: string): string {
  return createHash("sha256").update(text, "utf8").digest("hex");
}

function canonicalJson(value: unknown): string {
  if (value === null) return "null";
  switch (typeof value) {
    case "string": case "number": case "boolean":
      return JSON.stringify(value);
    case "object": {
      if (Array.isArray(value)) return `[${value.map((entry: unknown) => canonicalJson(entry)).join(",")}]`;
      const entries: [string, unknown][] = Object.entries(value);
      return `{${entries.filter(([, entry]) => entry !== undefined)
        .sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
        .map(([key, entry]) => `${JSON.stringify(key)}:${canonicalJson(entry)}`).join(",")}}`;
    }
    default:
      throw new Error("Embedding identity requires JSON semantic values");
  }
}

function identitySource(source: OnnxModelSource) {
  switch (source.kind) {
    case "local": return { kind: source.kind };
    case "huggingface": return { kind: source.kind, model: source.model, revision: source.revision.toLowerCase() };
    default: {
      const exhaustive: never = source;
      throw new Error(`Unsupported embedding source: ${exhaustive}`);
    }
  }
}

function normalizeArtifactPath(value: string): string {
  const path = value.replaceAll("\\", "/");
  const parts = path.split("/");
  if (path.startsWith("/") || path.includes(":") || path.includes("\0") || parts.includes("..")) {
    throw new Error("Artifact path must be a safe relative path");
  }
  const normalized = parts.filter(part => part !== "" && part !== ".").join("/");
  if (normalized === "") throw new Error("Artifact path must name a file");
  return normalized;
}

const digestSchema = z.string().regex(/^[a-f\d]{64}$/i).transform(value => value.toLowerCase());
const entrySchema = z.strictObject({
  path: z.string().transform(normalizeArtifactPath), sha256: digestSchema, bytes: z.number().int().nonnegative(),
});
const fileListSchema = z.array(entrySchema).min(1).max(128);
const positiveInteger = z.number().int().positive();
const spaceSchema = z.strictObject({
  version: z.literal(1), backend: z.literal("onnx"),
  artifactManifest: z.strictObject({ version: z.literal(1), source: z.unknown(), files: fileListSchema, digest: digestSchema }),
  dtype: z.enum(["fp32", "q8"]), nativeDimensions: positiveInteger, dimensions: positiveInteger,
  recipe: z.unknown(), formattingVersion: z.literal("qmd-onnx-format-v1"),
  tokenizerBehaviorVersion: z.literal("qmd-onnx-tokenizer-v1"),
  chunking: z.strictObject({ version: z.literal("qmd-onnx-chunk-v1"), maxInputTokens: positiveInteger,
    titleBudgetTokens: z.number().int().nonnegative(), overlapTokens: z.number().int().nonnegative() }),
});

/** Source is already parsed by the configuration boundary; this owns inventory hashing, not artifact IO. */
export function createArtifactManifest(source: OnnxModelSource, files: unknown): ArtifactManifest {
  const sorted = fileListSchema.parse(files).sort((a, b) => a.path < b.path ? -1 : a.path > b.path ? 1 : 0);
  const paths = new Set<string>();
  for (const file of sorted) {
    if (paths.has(file.path)) throw new Error(`Duplicate artifact path: ${file.path}`);
    paths.add(file.path);
  }
  const ownedFiles = Object.freeze(sorted.map(file => Object.freeze(file)));
  const digest = sha256(canonicalJson({ version: 1, source: identitySource(source), files: ownedFiles }));
  return Object.freeze({ version: 1, source: Object.freeze({ ...source }), files: ownedFiles, digest });
}

export function parseEmbeddingSpaceDescriptor(space: unknown): EmbeddingSpaceDescriptor {
  const parsed = spaceSchema.parse(space);
  const config = parseOnnxEmbeddingConfig({ backend: "onnx", source: parsed.artifactManifest.source,
    recipe: parsed.recipe, dtype: parsed.dtype, dimensions: parsed.dimensions }, { hostParallelism: 1 });
  const manifest = createArtifactManifest(config.source, parsed.artifactManifest.files);
  if (manifest.digest !== parsed.artifactManifest.digest) throw new Error("Embedding artifact manifest digest does not match its contents");
  if (parsed.nativeDimensions !== parsed.dimensions) throw new Error("Embedding dimensions require native output; reduction is not supported");
  const { chunking } = parsed;
  if (chunking.maxInputTokens > config.recipe.maxInputTokens
    || chunking.titleBudgetTokens > config.recipe.titleBudgetTokens
    || chunking.overlapTokens > config.recipe.overlapTokens
    || chunking.titleBudgetTokens + chunking.overlapTokens >= chunking.maxInputTokens) {
    throw new Error("Embedding chunk budgets must fit the declared recipe and leave positive body progress");
  }
  return Object.freeze({ ...parsed, recipe: Object.freeze({ ...config.recipe }),
    artifactManifest: manifest, chunking: Object.freeze({ ...parsed.chunking }) });
}

/** Canonical hash input omits local location; persist the complete descriptor separately for provenance. */
export function serializeEmbeddingSpace(space: unknown): string {
  const parsed = parseEmbeddingSpaceDescriptor(space);
  return canonicalJson({ ...parsed,
    artifactManifest: { ...parsed.artifactManifest, source: identitySource(parsed.artifactManifest.source) } });
}

/** Spec §8: preserve every SHA-256 bit; never reinterpret a legacy GGUF fingerprint. */
export function embeddingSpaceId(space: unknown): string {
  return `onnx-v1:${sha256(serializeEmbeddingSpace(space))}`;
}
