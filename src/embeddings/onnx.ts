import { prepareArtifactSnapshot } from "./artifacts.js";
import type { ArtifactIO, ArtifactSnapshot } from "./artifacts.js";
import { embeddingSpaceId } from "./identity.js";
import { onnxLoadOptions } from "./onnx-provider.js";
import { createOnnxSerialQueue } from "./onnx-queue.js";
import { planTokenBatches } from "./onnx-batching.js";
import { formatEmbeddingInput, measureEmbeddingInput, poolEmbeddingOutput, validateEmbeddingVectors } from "./recipe.js";
import type { EmbeddingTokenizer } from "./recipe.js";
import type { EmbeddingIdentity, RuntimeProvenance, EmbeddingCapability, ResolvedOnnxConfig, EmbeddingInput, EmbeddingSpaceDescriptor } from "./types.js";
import type { OnnxRuntimePort, OnnxRuntimeSession } from "./onnx-runtime-types.js";

class OnnxCapabilityError extends Error {}
type OnnxIdentity = Extract<EmbeddingIdentity, { kind: "onnx-v1" }>;
interface PreparedCapability {
  readonly snapshot: ArtifactSnapshot;
  readonly session: OnnxRuntimeSession;
  readonly identity: OnnxIdentity;
  readonly maxInputTokens: number;
  readonly provenance: RuntimeProvenance;
}
function assertNever(value: never): never { throw new Error(`Unsupported ONNX variant: ${String(value)}`); }

async function runAndPool(
  session: OnnxRuntimeSession, texts: readonly string[], recipe: ResolvedOnnxConfig["recipe"],
  dimensions: number | undefined, requestedDimensions: number | undefined, signal?: AbortSignal,
): Promise<readonly Float32Array[]> {
  let result: Awaited<ReturnType<OnnxRuntimeSession["run"]>>;
  try { result = await session.run(texts); }
  catch {
    if (signal?.aborted) throw signal.reason;
    throw new OnnxCapabilityError("ONNX inference failed");
  }
  let vectors: readonly Float32Array[] | undefined;
  let failed = false;
  let failure: unknown;
  try {
    signal?.throwIfAborted();
    vectors = poolEmbeddingOutput(result.output, result.mask, recipe);
    const nativeDimensions = dimensions ?? vectors[0]?.length;
    if (nativeDimensions === undefined) throw new OnnxCapabilityError("ONNX output has no measurable dimensions");
    validateEmbeddingVectors(vectors, texts.length, nativeDimensions, requestedDimensions);
  } catch (error) {
    failed = true;
    failure = signal?.aborted ? signal.reason : error instanceof OnnxCapabilityError
      ? error : new OnnxCapabilityError("ONNX output validation failed");
  }
  try { await result.release(); }
  catch {
    if (!failed && !signal?.aborted) {
      failed = true;
      failure = new OnnxCapabilityError("ONNX output cleanup failed");
    }
  }
  if (!failed && signal?.aborted) { failed = true; failure = signal.reason; }
  if (failed) throw failure;
  if (vectors === undefined) throw new OnnxCapabilityError("ONNX output unavailable");
  return vectors;
}

export function createOnnxEmbeddingCapability(
  config: ResolvedOnnxConfig, io: ArtifactIO, runtime: OnnxRuntimePort,
): EmbeddingCapability {
  let ready: PreparedCapability | undefined;
  let preparing: Promise<PreparedCapability> | undefined;
  let closePromise: Promise<void> | undefined;
  let closing = false;
  let active = 0;
  let drained: (() => void) | undefined;
  const serialQueue = createOnnxSerialQueue();
  const recipe = config.recipe;
  const modelLabel = (() => {
    switch (config.source.kind) {
      case "huggingface": return config.source.model;
      case "local": return "local ONNX model";
      default: return assertNever(config.source);
    }
  })();

  function begin(signal?: AbortSignal): () => void {
    if (closing) throw new OnnxCapabilityError("ONNX embedding capability is closing");
    signal?.throwIfAborted();
    active++;
    let finished = false;
    return () => {
      if (finished) return;
      finished = true;
      active--;
      if (active === 0) { drained?.(); drained = undefined; }
    };
  }

  async function initialize(): Promise<PreparedCapability> {
    // Reject an unsupported requested target before artifact IO or acquisition.
    const options = onnxLoadOptions(config, runtime);
    const snapshot = await prepareArtifactSnapshot(config, io);
    let session: OnnxRuntimeSession | undefined;
    try {
      if (snapshot.profile.kind !== "bert-encoder-v1" || recipe.maxInputTokens > snapshot.profile.maxInputTokens) {
        throw new OnnxCapabilityError("Recipe input budget exceeds the selected model context");
      }
      const maxInputTokens = Math.min(recipe.maxInputTokens, config.maxBatchTokens);
      if (recipe.titleBudgetTokens + recipe.overlapTokens >= maxInputTokens) {
        throw new OnnxCapabilityError("Effective ONNX input budget leaves no document body capacity");
      }
      let initializedMode = config.device;
      let cpuRetryReason: string | undefined;
      try { session = await runtime.load(snapshot, options); }
      catch (error) {
        let providerFailure = false;
        if (config.device !== "cpu") {
          try { providerFailure = runtime.isProviderInitializationError(error); }
          catch { providerFailure = false; }
        }
        if (!providerFailure) throw new OnnxCapabilityError("ONNX model initialization failed");
        if (config.fallback !== "cpu") {
          throw new OnnxCapabilityError("ONNX provider initialization failed; set fallback to cpu to allow a CPU retry");
        }
        cpuRetryReason = "provider-initialization-failed";
        runtime.diagnostic({ code: "ONNX_CPU_RETRY", requestedDevice: config.device, reason: cpuRetryReason });
        try { session = await runtime.load(snapshot, onnxLoadOptions(config, runtime, "cpu")); }
        catch { throw new OnnxCapabilityError("ONNX CPU fallback initialization failed"); }
        initializedMode = "cpu";
      }
      if (session === undefined) throw new OnnxCapabilityError("ONNX runtime did not initialize");
      const emptyDocument = await formatEmbeddingInput({ kind: "document", title: "", text: "" }, recipe, session.tokenizer);
      const documentOverhead = await session.tokenizer.encode(emptyDocument, true);
      if (documentOverhead.length + recipe.overlapTokens >= maxInputTokens) {
        throw new OnnxCapabilityError("Document prefix and overlap leave no body-token capacity");
      }
      const probe = await formatEmbeddingInput({ kind: "query", text: "x" }, recipe, session.tokenizer);
      const probeTokens = await session.tokenizer.encode(probe, true);
      if (probeTokens.length > maxInputTokens) throw new OnnxCapabilityError("ONNX query prefix exceeds the configured token budget");
      const vectors = await runAndPool(session, [probe], recipe, undefined, undefined);
      const vector = vectors[0];
      if (vector === undefined) throw new OnnxCapabilityError("ONNX dimension probe returned no vector");
      const dimensions = vector.length;
      if (config.dimensions !== undefined && config.dimensions !== dimensions) {
        throw new OnnxCapabilityError("Requested ONNX dimensions do not match measured native output");
      }
      validateEmbeddingVectors(vectors, 1, dimensions, config.dimensions);
      const descriptor: EmbeddingSpaceDescriptor = Object.freeze({
        version: 1, backend: "onnx", artifactManifest: snapshot.manifest, dtype: config.dtype,
        nativeDimensions: dimensions, dimensions, recipe,
        formattingVersion: "qmd-onnx-format-v1", tokenizerBehaviorVersion: "qmd-onnx-tokenizer-v1",
        chunking: Object.freeze({ version: "qmd-onnx-chunk-v1", maxInputTokens,
          titleBudgetTokens: recipe.titleBudgetTokens, overlapTokens: recipe.overlapTokens }),
      });
      const identity: OnnxIdentity = Object.freeze({ kind: "onnx-v1", id: embeddingSpaceId(descriptor), descriptor });
      const provenance: RuntimeProvenance = Object.freeze({ requestedDevice: config.device, initializedMode,
        ...(cpuRetryReason === undefined ? {} : { cpuRetryReason }), graphNodePlacement: "unknown" });
      return { snapshot, session, identity, maxInputTokens, provenance };
    } catch (error) {
      let cleanupFailed = false;
      if (session !== undefined) { try { await session.close(); } catch { cleanupFailed = true; } }
      try { await snapshot.release(); } catch { cleanupFailed = true; }
      if (error instanceof OnnxCapabilityError) throw error;
      throw new OnnxCapabilityError(cleanupFailed ? "ONNX preflight failed and cleanup was incomplete" : "ONNX preflight failed");
    }
  }

  function ensurePrepared(): Promise<PreparedCapability> {
    if (ready !== undefined) return Promise.resolve(ready);
    if (preparing === undefined) {
      preparing = initialize().then(state => { ready = state; return state; }).catch(error => {
        preparing = undefined;
        throw error;
      });
    }
    return preparing;
  }

  async function prepare(signal?: AbortSignal): Promise<EmbeddingIdentity> {
    const finish = begin(signal);
    try { const state = await ensurePrepared(); signal?.throwIfAborted(); return state.identity; }
    finally { finish(); }
  }

  async function withTokenizer<T>(operation: (tokenizer: EmbeddingTokenizer) => Promise<T>, message: string, signal?: AbortSignal): Promise<T> {
    const finish = begin(signal);
    try {
      const state = await ensurePrepared();
      const result = await serialQueue.run(() => operation(state.session.tokenizer), signal);
      signal?.throwIfAborted();
      return result;
    } catch (error) {
      if (signal?.aborted) throw signal.reason;
      if (error instanceof OnnxCapabilityError) throw error;
      throw new OnnxCapabilityError(message);
    } finally { finish(); }
  }
  const encode = (text: string) => withTokenizer(async tokenizer => Object.freeze([...await tokenizer.encode(text, true)]), "ONNX tokenization failed");
  const decode = (tokens: readonly number[]) => withTokenizer(tokenizer => tokenizer.decode(tokens), "ONNX detokenization failed");
  const measureTokens: NonNullable<EmbeddingCapability["measureTokens"]> = (input, signal) =>
    withTokenizer(tokenizer => measureEmbeddingInput(input, recipe, tokenizer), "ONNX token measurement failed", signal);

  async function embed(inputs: readonly EmbeddingInput[], signal?: AbortSignal): Promise<readonly Float32Array[]> {
    const finish = begin(signal);
    try {
      if (inputs.length === 0) return Object.freeze([]);
      const state = await ensurePrepared();
      signal?.throwIfAborted();
      return await serialQueue.run(async () => {
        const texts: string[] = [], lengths: number[] = [];
        for (const input of inputs) {
          signal?.throwIfAborted();
          const text = await formatEmbeddingInput(input, recipe, state.session.tokenizer);
          const tokens = await state.session.tokenizer.encode(text, true);
          if (tokens.length > state.maxInputTokens || tokens.length > config.maxBatchTokens) {
            throw new OnnxCapabilityError("ONNX input exceeds the total token budget; chunk documents or shorten queries");
          }
          texts.push(text); lengths.push(tokens.length);
        }
        const batches = planTokenBatches(lengths, { maxItems: config.batchSize,
          maxInputTokens: state.maxInputTokens, maxBatchTokens: config.maxBatchTokens });
        const ordered: (Float32Array | undefined)[] = inputs.map(() => undefined);
        for (const batch of batches) {
          signal?.throwIfAborted();
          const batchTexts: string[] = [];
          for (const index of batch) {
            const text = texts[index];
            if (text === undefined) throw new OnnxCapabilityError("ONNX batch index is invalid");
            batchTexts.push(text);
          }
          const vectors = await runAndPool(state.session, batchTexts, recipe,
            state.identity.descriptor.nativeDimensions, config.dimensions, signal);
          for (let index = 0; index < batch.length; index++) {
            const inputIndex = batch[index], vector = vectors[index];
            if (inputIndex === undefined || vector === undefined) throw new OnnxCapabilityError("ONNX batch output count changed");
            ordered[inputIndex] = vector;
          }
        }
        signal?.throwIfAborted();
        return Object.freeze(ordered.map(vector => {
          if (vector === undefined) throw new OnnxCapabilityError("ONNX result order is incomplete");
          return vector;
        }));
      }, signal);
    } finally { finish(); }
  }

  function close(): Promise<void> {
    if (closePromise !== undefined) return closePromise;
    closing = true;
    serialQueue.stop(new OnnxCapabilityError("ONNX embedding capability is closing"));
    const wait = active === 0 ? Promise.resolve() : new Promise<void>(resolve => { drained = resolve; });
    closePromise = wait.then(async () => {
      const state = ready;
      ready = undefined;
      if (state === undefined) return;
      let cleanupFailed = false;
      try { await state.session.close(); } catch { cleanupFailed = true; }
      try { await state.snapshot.release(); } catch { cleanupFailed = true; }
      if (cleanupFailed) throw new OnnxCapabilityError("ONNX capability cleanup failed");
    });
    return closePromise;
  }

  return Object.freeze({ config, modelLabel, get provenance() { return ready?.provenance; }, prepare, encode, decode, measureTokens, embed, close });
}
