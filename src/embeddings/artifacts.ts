import type { ArtifactIO } from "./artifact-io.js";
import type { ArtifactManifest, ResolvedOnnxConfig } from "./types.js";
import type { EmbeddingExportProfile } from "./artifact-profile.js";
import type { ArtifactFileReader } from "./artifact-local.js";
import type { ArtifactCacheEntry, ArtifactCacheKey } from "./artifact-cache.js";
import type { ArtifactStage, StagedArtifact } from "./artifact-files.js";
import { posix } from "node:path";
import { createArtifactManifest } from "./identity.js";
import { inspectOnnxExternalData } from "./onnx-graph.js";
import type { OnnxExternalData } from "./onnx-graph.js";
import { parseEmbeddingExportProfile } from "./artifact-profile.js";
export type { ArtifactIO } from "./artifact-io.js";

export interface ArtifactSnapshot {
  readonly directory: string;
  readonly manifest: ArtifactManifest;
  readonly profile: EmbeddingExportProfile;
  release(): Promise<void>;
}
class ArtifactPreparationError extends Error {}
type ResolvedSource = { readonly kind: "local"; readonly reader: ArtifactFileReader }
  | { readonly kind: "huggingface"; readonly reader: ArtifactFileReader; readonly key: ArtifactCacheKey; readonly cached?: ArtifactCacheEntry };
function assertNever(_value: never): never { throw new ArtifactPreparationError("Unsupported ONNX artifact variant"); }
async function resolveSource(config: ResolvedOnnxConfig, io: ArtifactIO, signal?: AbortSignal): Promise<ResolvedSource> {
  const source = config.source;
  switch (source.kind) {
    case "local": return { kind: "local", reader: await io.local(source.path) };
    case "huggingface": {
      const key: ArtifactCacheKey = { source, dtype: config.dtype }, cached = await io.cached(key, signal);
      signal?.throwIfAborted();
      if (cached !== undefined) {
        const origin = cached.manifest.source;
        if (origin.kind !== "huggingface" || origin.model !== source.model || origin.revision.toLowerCase() !== source.revision.toLowerCase()) {
          throw new ArtifactPreparationError("Cached artifacts do not match the configured pinned source");
        }
        return { kind: "huggingface", key, cached, reader: cached };
      }
      if (!config.allowDownload) throw new ArtifactPreparationError("ONNX artifacts are not cached; offline mode forbids acquisition");
      return { kind: "huggingface", key, reader: { source: (path, active) => io.download({ source, path }, active) } };
    }
    default: return assertNever(source);
  }
}
function selectedGraph(dtype: ResolvedOnnxConfig["dtype"]): string {
  switch (dtype) {
    case "fp32": return "onnx/model.onnx";
    case "q8": return "onnx/model_quantized.onnx";
    default: return assertNever(dtype);
  }
}
async function readJson(stage: ArtifactStage, path: string): Promise<unknown> {
  try { return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(await stage.readMetadata(path))); }
  catch { throw new ArtifactPreparationError("Invalid or oversized ONNX artifact JSON metadata"); }
}
export async function prepareArtifactSnapshot(
  config: ResolvedOnnxConfig, io: ArtifactIO, signal?: AbortSignal,
): Promise<ArtifactSnapshot> {
  signal?.throwIfAborted();
  let stage: ArtifactStage | undefined, ready = false, failed = false;
  try {
    const source = await resolveSource(config, io, signal);
    signal?.throwIfAborted();
    const owner = await io.stage(); stage = owner;
    const files = new Map<string, StagedArtifact>();
    let bytes = 0;
    async function acquire(path: string, metadata = false): Promise<StagedArtifact> {
      signal?.throwIfAborted();
      const existing = files.get(path);
      if (existing !== undefined) return existing;
      const remaining = 16 * 1024 ** 3 - bytes;
      if (remaining <= 0) throw new ArtifactPreparationError("ONNX artifact inventory exceeds the 16 GiB snapshot limit");
      const input = await source.reader.source(path, signal);
      signal?.throwIfAborted();
      if (input === undefined) throw new ArtifactPreparationError(`Missing selected ONNX artifact: ${path}`);
      const file = await owner.put({ path, source: input, maxBytes: Math.min(remaining, metadata ? 16 * 1024 ** 2 : 8 * 1024 ** 3) }, signal);
      files.set(path, file); bytes += file.entry.bytes;
      return file;
    }
    for (const path of ["config.json", "tokenizer_config.json", "tokenizer.json"]) await acquire(path, true);
    let profile: EmbeddingExportProfile;
    try {
      profile = parseEmbeddingExportProfile(await readJson(owner, "config.json"), await readJson(owner, "tokenizer_config.json"), await readJson(owner, "tokenizer.json"));
    } catch (error) {
      if (error instanceof ArtifactPreparationError) throw error;
      throw new ArtifactPreparationError("Unsupported ONNX embedding export profile");
    }
    const graph = selectedGraph(config.dtype); await acquire(graph);
    let references: readonly OnnxExternalData[];
    try { references = await inspectOnnxExternalData(owner.graph(graph), signal); }
    catch { signal?.throwIfAborted(); throw new ArtifactPreparationError("Invalid or unsupported ONNX graph closure"); }
    for (const reference of references) {
      const file = await acquire(posix.join(posix.dirname(graph), reference.location));
      if (reference.offset > file.entry.bytes || (reference.length !== undefined && reference.length > file.entry.bytes - reference.offset)
        || (reference.checksum !== undefined && reference.checksum !== file.sha1)) {
        throw new ArtifactPreparationError("Invalid ONNX external data range or checksum");
      }
    }
    signal?.throwIfAborted();
    const manifest = createArtifactManifest(config.source, [...files.values()].map(file => file.entry));
    switch (source.kind) {
      case "huggingface":
        if (source.cached !== undefined && source.cached.manifest.digest !== manifest.digest) {
          throw new ArtifactPreparationError("Cached inventory differs from the selected ONNX export closure");
        }
        break;
      case "local": break;
      default: assertNever(source);
    }
    await owner.seal(manifest); signal?.throwIfAborted();
    switch (source.kind) {
      case "huggingface":
        if (source.cached === undefined) await io.publish(source.key, { directory: owner.directory, manifest }, signal);
        break;
      case "local": break;
      default: assertNever(source);
    }
    signal?.throwIfAborted(); ready = true;
    let releasing: Promise<void> | undefined;
    const release = () => releasing ??= Promise.resolve().then(() => owner.release()).catch(() => {
      throw new ArtifactPreparationError("ONNX artifact snapshot cleanup failed");
    });
    return Object.freeze({ directory: owner.directory, manifest, profile, release });
  } catch (error) {
    failed = true; signal?.throwIfAborted();
    if (error instanceof ArtifactPreparationError) throw error;
    throw new ArtifactPreparationError("ONNX artifact source, size or integrity validation failed");
  } finally {
    if (stage !== undefined && !ready) {
      try { await stage.release(); }
      catch { if (!failed) throw new ArtifactPreparationError("ONNX artifact staging cleanup failed"); }
    }
  }
}
