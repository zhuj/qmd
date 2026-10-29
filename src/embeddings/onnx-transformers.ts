import { createHash } from "node:crypto";
import { isAbsolute, join, resolve } from "node:path";
import { z } from "zod";
import { createArtifactFileReader } from "./artifact-local.js";
import { parseEmbeddingExportProfile } from "./artifact-profile.js";
import { createArtifactManifest } from "./identity.js";
import { createNativeEmbeddingSession } from "./onnx-native-session.js";
import { nativeMethod, nativeObject, nativeProperty } from "./onnx-native-values.js";
import { isOnnxProviderInitializationError } from "./onnx-provider-error.js";
import { assertTransformersEnvironment } from "./onnx-transformers-env.js";
import type { ArtifactSnapshot } from "./artifacts.js";
import type { OnnxRuntimePort } from "./onnx-runtime-types.js";
import type { OnnxLoadOptions } from "./onnx-runtime-types.js";
import type { EmbeddingTokenizer } from "./recipe.js";

const MAX_METADATA_BYTES = 16 * 1024 * 1024;
const CACHE_DIRECTORY = ".qmd-transformers-cache";
const configSchema = z.object({
  model_type: z.literal("bert"),
  hidden_size: z.number().int().positive(),
  max_position_embeddings: z.number().int().min(3),
});
const loadOptionsSchema = z.object({
  device: z.enum(["cpu", "cuda", "dml", "coreml"]),
  providers: z.array(z.enum(["cpu", "cuda", "dml", "coreml"])).length(1),
  dtype: z.enum(["fp32", "q8"]),
  intraOpNumThreads: z.number().int().positive(),
  sequential: z.literal(true),
  memoryPattern: z.boolean(),
  localFilesOnly: z.literal(true),
}).strict();

interface RuntimeConstructor { new (...args: unknown[]): unknown }
interface RuntimeModule {
  readonly env: unknown;
  readonly BertTokenizer: RuntimeConstructor;
  readonly BertModel: unknown;
  readonly Tensor: RuntimeConstructor;
}
class RuntimeAdapterError extends Error {}
class ProviderInitializationFailure extends Error {
  constructor() { super("ONNX execution provider initialization failed"); }
}
function isRuntimeConstructor(value: unknown): value is RuntimeConstructor {
  return typeof value === "function" && Object.hasOwn(value, "prototype");
}
function runtimeConstructor(value: unknown): RuntimeConstructor {
  if (!isRuntimeConstructor(value)) throw new RuntimeAdapterError("Unsupported Transformers.js runtime API");
  return value;
}
function readRuntimeModule(value: unknown): RuntimeModule {
  try {
    const module = nativeObject(value);
    const env = nativeProperty(module, "env");
    if (nativeProperty(env, "version") !== "4.3.1") throw new RuntimeAdapterError("Unsupported Transformers.js version");
    return {
      env,
      BertTokenizer: runtimeConstructor(nativeProperty(module, "BertTokenizer")),
      BertModel: nativeProperty(module, "BertModel"),
      Tensor: runtimeConstructor(nativeProperty(module, "Tensor")),
    };
  } catch (error) {
    if (error instanceof RuntimeAdapterError) throw error;
    throw new RuntimeAdapterError("Unsupported Transformers.js runtime API");
  }
}
function parseLoadOptions(value: unknown): OnnxLoadOptions {
  const parsed = loadOptionsSchema.safeParse(value);
  if (!parsed.success || parsed.data.providers[0] !== parsed.data.device
    || (parsed.data.device === "dml" && parsed.data.memoryPattern)) {
    throw new RuntimeAdapterError("Unsupported ONNX runtime options");
  }
  return parsed.data;
}
function selectedGraph(dtype: OnnxLoadOptions["dtype"]): string {
  switch (dtype) {
    case "fp32": return "onnx/model.onnx";
    case "q8": return "onnx/model_quantized.onnx";
    default: throw new RuntimeAdapterError("Unsupported ONNX dtype");
  }
}
async function readSnapshotJson(
  reader: Awaited<ReturnType<typeof createArtifactFileReader>>,
  snapshot: ArtifactSnapshot,
  path: string,
): Promise<unknown> {
  const entry = snapshot.manifest.files.find(file => file.path === path);
  if (entry === undefined || !Number.isSafeInteger(entry.bytes) || entry.bytes < 0 || entry.bytes > MAX_METADATA_BYTES) {
    throw new RuntimeAdapterError("ONNX snapshot metadata is missing or oversized");
  }
  const source = await reader.source(path);
  if (source === undefined || source.byteLength !== entry.bytes) throw new RuntimeAdapterError("ONNX snapshot metadata is unavailable");
  const chunks: Uint8Array[] = [];
  let size = 0;
  for await (const chunk of source.chunks()) {
    size += chunk.byteLength;
    if (size > entry.bytes || size > MAX_METADATA_BYTES) throw new RuntimeAdapterError("ONNX snapshot metadata is oversized");
    chunks.push(chunk.slice());
  }
  if (size !== entry.bytes) throw new RuntimeAdapterError("ONNX snapshot metadata is truncated");
  const bytes = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.byteLength; }
  if (createHash("sha256").update(bytes).digest("hex") !== entry.sha256) {
    throw new RuntimeAdapterError("ONNX snapshot metadata integrity mismatch");
  }
  try { return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes)); }
  catch { throw new RuntimeAdapterError("ONNX snapshot metadata is invalid"); }
}
async function readBertSnapshot(snapshot: ArtifactSnapshot, dtype: OnnxLoadOptions["dtype"]) {
  let stage = "snapshot path lookup";
  try {
  const snapshotDirectory = snapshot.directory;
  stage = typeof snapshotDirectory === "string" ? "snapshot path check" : "snapshot directory type";
  if (typeof snapshotDirectory !== "string" || !isAbsolute(snapshotDirectory)) throw new RuntimeAdapterError("ONNX snapshot path must be absolute");
  let manifest: ReturnType<typeof createArtifactManifest>;
  stage = "manifest parsing";
  try { manifest = createArtifactManifest(snapshot.manifest.source, snapshot.manifest.files); }
  catch { throw new RuntimeAdapterError("ONNX snapshot manifest is invalid"); }
  stage = "manifest identity";
  if (manifest.digest !== snapshot.manifest.digest || snapshot.profile.kind !== "bert-encoder-v1") {
    throw new RuntimeAdapterError("ONNX snapshot identity does not match its validated profile");
  }
  stage = "cache inventory";
  if (manifest.files.some(file => file.path === CACHE_DIRECTORY || file.path.startsWith(`${CACHE_DIRECTORY}/`))) {
    throw new RuntimeAdapterError("ONNX snapshot uses a reserved runtime cache path");
  }
  stage = "selected graph inventory";
  const graphPath = selectedGraph(dtype), graphEntry = manifest.files.find(file => file.path === graphPath);
  if (graphEntry === undefined) throw new RuntimeAdapterError("Selected ONNX graph is missing from the snapshot manifest");
  stage = "graph";
  const reader = await createArtifactFileReader(resolve(snapshot.directory));
  const graph = await reader.source(graphPath);
  if (graph === undefined || graph.byteLength !== graphEntry.bytes) throw new RuntimeAdapterError("Selected ONNX graph is unavailable");
  stage = "metadata";
  const configJson = await readSnapshotJson(reader, snapshot, "config.json");
  const tokenizerConfig = await readSnapshotJson(reader, snapshot, "tokenizer_config.json");
  const tokenizerJson = await readSnapshotJson(reader, snapshot, "tokenizer.json");
  let profile: ReturnType<typeof parseEmbeddingExportProfile>;
  let modelConfig: z.infer<typeof configSchema>;
  try {
    profile = parseEmbeddingExportProfile(configJson, tokenizerConfig, tokenizerJson);
    modelConfig = configSchema.parse(configJson);
  } catch { throw new RuntimeAdapterError("Unsupported ONNX embedding export profile"); }
  if (profile.kind !== snapshot.profile.kind || profile.hiddenSize !== snapshot.profile.hiddenSize
    || profile.maxInputTokens !== snapshot.profile.maxInputTokens || profile.tokenizerClass !== snapshot.profile.tokenizerClass) {
    throw new RuntimeAdapterError("ONNX snapshot profile changed after artifact validation");
  }
  return { directory: resolve(snapshot.directory), manifest, profile, modelConfig, tokenizerConfig, tokenizerJson };
  } catch (error) {
    if (error instanceof RuntimeAdapterError) throw error;
    throw new RuntimeAdapterError(`ONNX artifact snapshot validation failed during ${stage}`);
  }
}

export function createTransformersRuntime(
  loadModule?: () => Promise<unknown>,
): OnnxRuntimePort {
  const importModule = loadModule ?? (() => import("@huggingface/transformers"));
  let modulePromise: Promise<RuntimeModule> | undefined;
  async function getModule(): Promise<RuntimeModule> {
    if (modulePromise === undefined) {
      modulePromise = Promise.resolve().then(importModule).then(readRuntimeModule).catch(error => {
        modulePromise = undefined;
        if (error instanceof RuntimeAdapterError) throw error;
        throw new RuntimeAdapterError("ONNX runtime module is unavailable");
      });
    }
    return modulePromise;
  }
  async function load(snapshot: ArtifactSnapshot, rawOptions: OnnxLoadOptions) {
    let stage = "options";
    try {
      const options = parseLoadOptions(rawOptions);
      stage = "snapshot";
      const artifacts = await readBertSnapshot(snapshot, options.dtype);
      stage = "runtime module";
      const module = await getModule();
      stage = "environment";
      try { assertTransformersEnvironment(module.env); }
      catch { throw new RuntimeAdapterError("Transformers.js environment is incompatible"); }
      stage = "tokenizer";
      const tokenizer = new module.BertTokenizer(artifacts.tokenizerJson, artifacts.tokenizerConfig);
      const sessionOptions = Object.freeze({
        executionProviders: Object.freeze([...options.providers]),
        intraOpNumThreads: options.intraOpNumThreads,
        executionMode: "sequential",
        enableMemPattern: options.memoryPattern,
        freeDimensionOverrides: Object.freeze({}),
      });
      const modelOptions = Object.freeze({
        config: artifacts.modelConfig,
        cache_dir: join(artifacts.directory, CACHE_DIRECTORY),
        local_files_only: true,
        model_file_name: "model",
        subfolder: "onnx",
        device: options.device,
        dtype: options.dtype,
        use_external_data_format: false,
        session_options: sessionOptions,
      });
      const fromPretrained = nativeMethod(module.BertModel, "from_pretrained");
      stage = "model";
      let model: unknown;
      try { model = await fromPretrained(artifacts.directory, modelOptions); }
      catch (error) {
        if (isOnnxProviderInitializationError(error, options.device)) throw new ProviderInitializationFailure();
        throw new RuntimeAdapterError("ONNX model loading failed");
      }
      stage = "session contract";
      try {
        return createNativeEmbeddingSession({
          tokenizer,
          model,
          int64Tensor: (data, dims) => new module.Tensor("int64", data, [...dims]),
        });
      } catch {
        try { await nativeMethod(model, "dispose")(); } catch { /* Preserve admission failure over secondary cleanup. */ }
        throw new RuntimeAdapterError("Unsupported ONNX session contract");
      }
    } catch (error) {
      if (error instanceof ProviderInitializationFailure || error instanceof RuntimeAdapterError) throw error;
      throw new RuntimeAdapterError(`ONNX runtime loading failed during ${stage}`);
    }
  }
  return {
    platform: process.platform,
    arch: process.arch,
    load,
    isProviderInitializationError(error) {
      try { return error instanceof ProviderInitializationFailure; } catch { return false; }
    },
    diagnostic(event) {
      console.warn("[qmd] ONNX CPU fallback", {
        code: event.code,
        requestedDevice: event.requestedDevice,
        reason: "Requested ONNX provider was unavailable during initialization",
      });
    },
  };
}
