import { dirname, isAbsolute, resolve } from "path";
import { z } from "zod";
import type { OnnxEmbeddingConfig, ResolvedEmbeddingConfig, SearchStageRequest, SearchStagePolicy } from "./types.js";

const positiveInteger = z.number().int().positive();
const repositoryId = z.string().min(1).max(96).refine(value =>
  /^(?:[\w][\w.-]*\/)?[\w][\w.-]*$/.test(value)
  && !value.split("/").some(part => part.endsWith(".") || part.endsWith("-") || part.endsWith(".git"))
  && !value.includes("--") && !value.includes(".."), "Expected a Hugging Face repository ID, not a URL or path");
const sourceSchema = z.discriminatedUnion("kind", [
  z.strictObject({ kind: z.literal("huggingface"), model: repositoryId,
    revision: z.string().regex(/^[a-fA-F0-9]{40}$/, "Expected an immutable 40-hex commit revision") }),
  z.strictObject({ kind: z.literal("local"), path: z.string().min(1).refine(
    path => !path.includes("\0") && !/^[a-z][a-z\d+.-]*:\/\//i.test(path), "Expected a local filesystem path, not a URL",
  ) }),
]);
const recipeSchema = z.strictObject({
  version: z.literal("qmd-embedding-v1"),
  pooling: z.enum(["mean", "cls", "last-token", "sentence-output"]),
  normalization: z.enum(["none", "l2"]),
  queryPrefix: z.string(), documentPrefix: z.string(),
  maxInputTokens: positiveInteger,
  titleBudgetTokens: z.number().int().nonnegative(),
  overlapTokens: z.number().int().nonnegative(),
  clsTokenIndex: z.number().int().nonnegative().optional(),
  sentenceOutput: z.literal("sentence_embedding").optional(),
}).superRefine((recipe, ctx) => {
  if (recipe.titleBudgetTokens + recipe.overlapTokens >= recipe.maxInputTokens) {
    ctx.addIssue({ code: "custom", path: ["overlapTokens"], message: "Title and overlap budgets must leave positive document body progress" });
  }
  switch (recipe.pooling) {
    case "cls":
      if (recipe.clsTokenIndex === undefined || recipe.clsTokenIndex >= recipe.maxInputTokens) {
        ctx.addIssue({ code: "custom", path: ["clsTokenIndex"], message: "CLS requires a token index within maxInputTokens" });
      }
      break;
    case "mean": case "last-token": case "sentence-output":
      if (recipe.clsTokenIndex !== undefined) {
        ctx.addIssue({ code: "custom", path: ["clsTokenIndex"], message: "clsTokenIndex is only supported for CLS pooling" });
      }
      break;
    default: {
      const exhaustive: never = recipe.pooling;
      throw new Error(`Unsupported pooling: ${exhaustive}`);
    }
  }
  if ((recipe.pooling === "sentence-output") !== (recipe.sentenceOutput !== undefined)) {
    ctx.addIssue({ code: "custom", path: ["sentenceOutput"], message: "sentence-output pooling requires only the sentence_embedding output" });
  }
});

/** Parse only declared settings; model existence/output contracts belong to preparation. */
export function parseOnnxEmbeddingConfig(
  value: unknown, context: Pick<EmbeddingConfigContext, "configPath" | "hostParallelism">,
): OnnxEmbeddingConfig {
  const parsed = z.strictObject({
    backend: z.literal("onnx"), source: sourceSchema, recipe: recipeSchema,
    device: z.enum(["cpu", "cuda", "dml", "coreml"]).optional(),
    fallback: z.enum(["error", "cpu"]).optional(),
    dtype: z.enum(["fp32", "q8"]).optional(),
    dimensions: positiveInteger.optional(), allowDownload: z.boolean().optional(),
    intraOpNumThreads: positiveInteger.max(context.hostParallelism).optional(),
    batchSize: positiveInteger.max(32).optional(),
    maxBatchTokens: positiveInteger.max(65536).optional(),
  }).parse(value);
  switch (parsed.source.kind) {
    case "local":
      if (!isAbsolute(parsed.source.path) && context.configPath === undefined) {
        throw new Error("Inline ONNX local source.path must be absolute; relative paths require a config file path");
      }
      return { ...parsed, source: { kind: "local", path: resolve(
        context.configPath === undefined ? "." : dirname(context.configPath), parsed.source.path,
      ) } };
    case "huggingface":
      return parsed;
    default: {
      const exhaustive: never = parsed.source;
      throw new Error(`Unsupported ONNX source: ${exhaustive}`);
    }
  }
}

export interface EmbeddingConfigContext {
  readonly configPath?: string;
  readonly legacyEnv?: string;
  readonly defaultGgufModel: string;
  readonly hostParallelism: number;
}

export function resolveEmbeddingConfig(value: unknown, context: EmbeddingConfigContext): ResolvedEmbeddingConfig {
  if (value === undefined || typeof value === "string") {
    return Object.freeze({ backend: "gguf", model: value || context.legacyEnv || context.defaultGgufModel });
  }
  const config = parseOnnxEmbeddingConfig(value, context);
  return Object.freeze({
    ...config, source: Object.freeze(config.source), recipe: Object.freeze(config.recipe),
    device: config.device ?? "cpu", fallback: config.fallback ?? "error", dtype: config.dtype ?? "fp32",
    allowDownload: config.allowDownload ?? false, intraOpNumThreads: config.intraOpNumThreads ?? 1,
    batchSize: config.batchSize ?? 1, maxBatchTokens: config.maxBatchTokens ?? 4096,
  });
}

export function resolveSearchStagePolicy(config: ResolvedEmbeddingConfig, request: SearchStageRequest): SearchStagePolicy {
  const cpuOnly = config.backend === "onnx" && config.device === "cpu";
  if (cpuOnly && (request.expand === true || request.rerank === true)) {
    throw new Error("CPU-only ONNX policy disables query expansion and neural reranking; omit these stages or explicitly disable them");
  }
  return { expand: request.expand ?? !cpuOnly, rerank: request.rerank ?? !cpuOnly };
}
