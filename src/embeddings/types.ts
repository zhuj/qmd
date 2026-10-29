export type OnnxModelSource =
  | Readonly<{ kind: "huggingface"; model: string; revision: string }>
  | Readonly<{ kind: "local"; path: string }>;

export interface EmbeddingRecipe {
  readonly version: "qmd-embedding-v1";
  readonly pooling: "mean" | "cls" | "last-token" | "sentence-output";
  readonly normalization: "none" | "l2";
  readonly queryPrefix: string;
  readonly documentPrefix: string;
  readonly maxInputTokens: number;
  readonly titleBudgetTokens: number;
  readonly overlapTokens: number;
  readonly clsTokenIndex?: number;
  readonly sentenceOutput?: "sentence_embedding";
  readonly reduction?: Readonly<{ profile: string }>;
}

export interface OnnxEmbeddingConfig {
  readonly backend: "onnx";
  readonly source: OnnxModelSource;
  readonly recipe: EmbeddingRecipe;
  readonly device?: "cpu" | "cuda" | "dml" | "coreml";
  readonly fallback?: "error" | "cpu";
  readonly dtype?: "fp32" | "q8";
  readonly dimensions?: number;
  readonly allowDownload?: boolean;
  readonly intraOpNumThreads?: number;
  readonly batchSize?: number;
  readonly maxBatchTokens?: number;
}

export type ResolvedOnnxConfig = Readonly<OnnxEmbeddingConfig & {
  device: "cpu" | "cuda" | "dml" | "coreml";
  fallback: "error" | "cpu";
  dtype: "fp32" | "q8";
  allowDownload: boolean;
  intraOpNumThreads: number;
  batchSize: number;
  maxBatchTokens: number;
}>;
export type ResolvedEmbeddingConfig = Readonly<{ backend: "gguf"; model: string }> | ResolvedOnnxConfig;
export interface ArtifactEntry { readonly path: string; readonly sha256: string; readonly bytes: number }
export interface ArtifactManifest {
  readonly version: 1;
  readonly source: OnnxModelSource;
  readonly files: readonly ArtifactEntry[];
  readonly digest: string;
}
export interface EmbeddingSpaceDescriptor {
  readonly version: 1;
  readonly backend: "onnx";
  readonly artifactManifest: ArtifactManifest;
  readonly dtype: "fp32" | "q8";
  readonly nativeDimensions: number;
  readonly dimensions: number;
  readonly recipe: EmbeddingRecipe;
  readonly formattingVersion: "qmd-onnx-format-v1";
  readonly tokenizerBehaviorVersion: "qmd-onnx-tokenizer-v1";
  readonly chunking: Readonly<{ version: "qmd-onnx-chunk-v1"; maxInputTokens: number;
    titleBudgetTokens: number; overlapTokens: number }>;
}
export type EmbeddingIdentity =
  | Readonly<{ kind: "legacy-gguf"; model: string; fingerprint: string; dimensions: number }>
  | Readonly<{ kind: "onnx-v1"; id: string; descriptor: EmbeddingSpaceDescriptor }>;
export type EmbeddingInput =
  | Readonly<{ kind: "query"; text: string }>
  | Readonly<{ kind: "document"; text: string; title: string }>;
export interface EmbeddingTokenMeasurement {
  readonly text: number;
  readonly total: number;
}
export interface RuntimeProvenance {
  readonly requestedDevice: "cpu" | "cuda" | "dml" | "coreml";
  readonly initializedMode: "cpu" | "cuda" | "dml" | "coreml";
  readonly cpuRetryReason?: string;
  readonly graphNodePlacement: "unknown";
}
export interface EmbeddingCapability {
  readonly config: ResolvedEmbeddingConfig;
  readonly modelLabel: string;
  readonly provenance: RuntimeProvenance | undefined;
  prepare(signal?: AbortSignal): Promise<EmbeddingIdentity>;
  encode(text: string): Promise<readonly number[]>;
  measureTokens?(input: EmbeddingInput, signal?: AbortSignal): Promise<EmbeddingTokenMeasurement>;
  decode(tokens: readonly number[]): Promise<string>;
  embed(inputs: readonly EmbeddingInput[], signal?: AbortSignal): Promise<readonly Float32Array[]>;
  close(): Promise<void>;
}
export interface SearchStageRequest { readonly expand?: boolean; readonly rerank?: boolean }
export interface SearchStagePolicy { readonly expand: boolean; readonly rerank: boolean }
