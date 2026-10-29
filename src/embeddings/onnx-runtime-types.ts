import type { ArtifactSnapshot } from "./artifacts.js";
import type { EmbeddingTokenizer, EmbeddingModelOutput } from "./recipe.js";

export interface OnnxRuntimeSession {
  readonly tokenizer: EmbeddingTokenizer;
  run(texts: readonly string[]): Promise<{
    readonly output: EmbeddingModelOutput;
    readonly mask: readonly (readonly number[])[];
    release(): Promise<void>;
  }>;
  close(): Promise<void>;
}
export interface OnnxLoadOptions {
  readonly device: "cpu" | "cuda" | "dml" | "coreml";
  readonly providers: readonly ("cpu" | "cuda" | "dml" | "coreml")[];
  readonly dtype: "fp32" | "q8";
  readonly intraOpNumThreads: number;
  readonly sequential: boolean;
  readonly memoryPattern: boolean;
  readonly localFilesOnly: true;
}
export interface OnnxRuntimePort {
  readonly platform: NodeJS.Platform;
  readonly arch: string;
  load(snapshot: ArtifactSnapshot, options: OnnxLoadOptions): Promise<OnnxRuntimeSession>;
  isProviderInitializationError(error: unknown): boolean;
  diagnostic(event: { readonly code: "ONNX_CPU_RETRY"; readonly requestedDevice: string; readonly reason: string }): void;
}
