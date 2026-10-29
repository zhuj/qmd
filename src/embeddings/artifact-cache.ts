import type { ArtifactManifest, OnnxModelSource } from "./types.js";
import type { ArtifactFileReader } from "./artifact-local.js";
import { createHash } from "node:crypto";
import { constants } from "node:fs";
import { chmod, copyFile, lstat, mkdir, mkdtemp, rename, rm, writeFile } from "node:fs/promises";
import { dirname, join } from "node:path";
import { z } from "zod";
import { createArtifactFileReader } from "./artifact-local.js";
import { validateArtifactPath } from "./artifact-path.js";
import { createArtifactManifest } from "./identity.js";

export interface ArtifactCacheKey {
  readonly source: Extract<OnnxModelSource, { kind: "huggingface" }>;
  readonly dtype: "fp32" | "q8";
}
export interface ArtifactCacheSnapshot {
  readonly directory: string;
  readonly manifest: ArtifactManifest;
}
export interface ArtifactCacheEntry extends ArtifactFileReader { readonly manifest: ArtifactManifest }
export interface ArtifactCache {
  find(key: ArtifactCacheKey, signal?: AbortSignal): Promise<ArtifactCacheEntry | undefined>;
  publish(key: ArtifactCacheKey, snapshot: ArtifactCacheSnapshot, signal?: AbortSignal): Promise<void>;
}
function ownKey(key: ArtifactCacheKey): ArtifactCacheKey {
  return { source: { kind: "huggingface", model: key.source.model, revision: key.source.revision.toLowerCase() }, dtype: key.dtype };
}
function slotName(key: ArtifactCacheKey): string {
  return createHash("sha256").update(JSON.stringify({ layout: "qmd-onnx-cache-v1", source: key.source, dtype: key.dtype })).digest("hex");
}
function parseManifest(value: unknown, key: ArtifactCacheKey): ArtifactManifest {
  const parsed = z.strictObject({ version: z.literal(1), files: z.unknown(),
    source: z.strictObject({ kind: z.literal("huggingface"), model: z.literal(key.source.model),
      revision: z.string().transform(text => text.toLowerCase()).pipe(z.literal(key.source.revision)) }),
    digest: z.string().regex(/^[a-f\d]{64}$/i).transform(text => text.toLowerCase()),
  }).parse(value);
  const manifest = createArtifactManifest(key.source, parsed.files), paths = new Set<string>();
  if (manifest.digest !== parsed.digest) throw new Error("Cached manifest digest mismatch");
  for (const file of manifest.files) {
    validateArtifactPath(file.path);
    const folded = file.path.toLowerCase();
    if (folded === ".qmd-manifest.json" || paths.has(folded)) throw new Error("Unsafe cached artifact inventory");
    paths.add(folded);
  }
  return manifest;
}
async function directoryExists(path: string): Promise<boolean> {
  try {
    if (!(await lstat(path)).isDirectory()) throw new Error("Unsafe cache directory");
    return true;
  } catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") return false;
    throw error;
  }
}
export function createArtifactCache(directory: string): ArtifactCache {
  const models = join(directory, "models");
  const cache: ArtifactCache = {
    async find(input, signal) {
      signal?.throwIfAborted();
      const key = ownKey(input), target = join(models, slotName(key));
      try {
        for (const path of [directory, models, target]) {
          const exists = await directoryExists(path);
          signal?.throwIfAborted();
          if (!exists) return undefined;
          await createArtifactFileReader(path);
        }
        const reader = await createArtifactFileReader(target), metadata = await reader.source(".qmd-manifest.json", signal);
        if (metadata === undefined || metadata.byteLength === undefined || metadata.byteLength > 1048576) throw new Error("Invalid cached metadata size");
        const buffers: Uint8Array[] = [];
        for await (const bytes of metadata.chunks()) buffers.push(bytes);
        signal?.throwIfAborted();
        const value: unknown = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(Buffer.concat(buffers)));
        const manifest = parseManifest(value, key), files = new Map(manifest.files.map(file => [file.path, file]));
        return Object.freeze({ manifest,
          async source(path: string, sourceSignal?: AbortSignal) {
            const active = signal && sourceSignal ? AbortSignal.any([signal, sourceSignal]) : signal ?? sourceSignal;
            active?.throwIfAborted(); validateArtifactPath(path);
            const expected = files.get(path);
            if (expected === undefined) return undefined;
            const source = await reader.source(path, active);
            if (source === undefined) return undefined;
            if (source.byteLength !== expected.bytes) throw new Error("Cached ONNX artifact size mismatch");
            // The consuming owned stage hashes actual bytes before native use.
            return Object.freeze({ ...source, expectedSha256: expected.sha256 });
          },
        });
      } catch {
        signal?.throwIfAborted();
        throw new Error("Cached ONNX snapshot is invalid or unavailable");
      }
    },
    async publish(input, snapshot, signal) {
      signal?.throwIfAborted();
      const key = ownKey(input), sourceDirectory = snapshot.directory;
      let pending: string | undefined, failed = false;
      try {
        const manifest = parseManifest(snapshot.manifest, key), text = JSON.stringify(manifest);
        if (Buffer.byteLength(text) > 1048576) throw new Error("Cache manifest exceeds metadata limit");
        await mkdir(directory, { recursive: true, mode: 0o700 });
        await createArtifactFileReader(directory);
        await mkdir(models, { mode: 0o700 }).catch(error => {
          if (!(error instanceof Error && "code" in error && error.code === "EEXIST")) throw error;
        });
        await createArtifactFileReader(models);
        // Input is the coordinator's sealed private snapshot, not a mutable
        // user model directory. Never transfer the caller's stage ownership.
        await createArtifactFileReader(sourceDirectory);
        pending = await mkdtemp(join(models, "qmd-publish-"));
        for (const file of manifest.files) {
          signal?.throwIfAborted();
          const target = join(pending, file.path);
          await mkdir(dirname(target), { recursive: true, mode: 0o700 });
          await copyFile(join(sourceDirectory, file.path), target, constants.COPYFILE_EXCL);
          await chmod(target, 0o400);
        }
        await writeFile(join(pending, ".qmd-manifest.json"), text, { flag: "wx", mode: 0o400 });
        signal?.throwIfAborted();
        try {
          await rename(pending, join(models, slotName(key)));
          pending = undefined;
        } catch {
          const existing = await cache.find(key, signal);
          if (existing?.manifest.digest !== manifest.digest) throw new Error("Conflicting immutable cache contents");
        }
      } catch {
        failed = true;
        signal?.throwIfAborted();
        throw new Error("ONNX cache publication failed or conflicts with an existing snapshot");
      } finally {
        if (pending !== undefined) {
          try { await rm(pending, { recursive: true, force: true }); }
          catch { if (!failed) throw new Error("ONNX cache pending-directory cleanup failed"); }
        }
      }
    },
  };
  return Object.freeze(cache);
}
