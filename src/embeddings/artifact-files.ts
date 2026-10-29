import type { ArtifactEntry, ArtifactManifest } from "./types.js";
import type { OnnxGraphFile } from "./onnx-graph.js";
import { createHash } from "node:crypto";
import { chmod, mkdir, mkdtemp, open, rm, writeFile, type FileHandle } from "node:fs/promises";
import { join } from "node:path";
import { createArtifactManifest } from "./identity.js";
import { validateArtifactPath } from "./artifact-path.js";

export interface ArtifactSource {
  readonly byteLength?: number;
  readonly expectedSha256?: string;
  readonly expectedGitSha1?: string;
  chunks(signal?: AbortSignal): AsyncIterable<Uint8Array>;
}
export interface ArtifactTransfer {
  readonly path: string;
  readonly source: ArtifactSource;
  readonly maxBytes: number;
}
export interface StagedArtifact { readonly entry: ArtifactEntry; readonly sha1: string }
export interface ArtifactStage {
  readonly directory: string;
  put(transfer: ArtifactTransfer, signal?: AbortSignal): Promise<StagedArtifact>;
  graph(path: string): OnnxGraphFile;
  readMetadata(path: string): Promise<Uint8Array>;
  seal(manifest: ArtifactManifest): Promise<void>;
  release(): Promise<void>;
}

export async function createArtifactStage(root: string): Promise<ArtifactStage> {
  await mkdir(root, { recursive: true, mode: 0o700 });
  const directory = await mkdtemp(join(root, "qmd-onnx-"));
  const files = new Map<string, StagedArtifact>();
  const names = new Set<string>();
  const directories = new Set([directory]);
  const pending = new Set<Promise<StagedArtifact>>();
  const controller = new AbortController();
  let releasing: Promise<void> | undefined;
  let sealed = false;

  function pathFor(path: string): string {
    validateArtifactPath(path);
    if (path.toLowerCase() === ".qmd-manifest.json") throw new Error("Reserved artifact path");
    return join(directory, path);
  }
  function writable(): void {
    if (sealed || releasing) throw new Error("Artifact stage is sealed or closing");
  }
  function staged(path: string): StagedArtifact {
    if (releasing) throw new Error("Artifact stage is closing");
    const result = files.get(path);
    if (!result) throw new Error("Missing staged artifact");
    return result;
  }
  async function write(transfer: ArtifactTransfer, signal?: AbortSignal): Promise<StagedArtifact> {
    writable();
    const { path, source, maxBytes } = transfer;
    const target = pathFor(path);
    const signalForWrite = signal ? AbortSignal.any([signal, controller.signal]) : controller.signal;
    signalForWrite.throwIfAborted();
    if (names.has(path.toLowerCase())) throw new Error("Duplicate artifact path");
    if (names.size >= 128) throw new Error("Artifact inventory limit exceeded");
    if (!Number.isSafeInteger(maxBytes) || maxBytes < 0 || (source.byteLength !== undefined
      && (!Number.isSafeInteger(source.byteLength) || source.byteLength < 0 || source.byteLength > maxBytes))) {
      throw new Error("Invalid artifact size/length limit");
    }
    if (source.expectedGitSha1 && source.byteLength === undefined) throw new Error("Git artifact integrity requires declared length");
    names.add(path.toLowerCase());
    let handle: FileHandle;
    try {
      let parent = directory;
      for (const part of path.split("/").slice(0, -1)) {
        parent = join(parent, part);
        await mkdir(parent, { recursive: true, mode: 0o700 });
        directories.add(parent);
      }
      handle = await open(target, "wx", 0o600);
    } catch (error) {
      names.delete(path.toLowerCase());
      throw error;
    }
    const sha256 = createHash("sha256");
    const sha1 = createHash("sha1");
    const git = source.expectedGitSha1 ? createHash("sha1").update(`blob ${source.byteLength}\0`) : undefined;
    let size = 0;
    let complete = false;
    try {
      for await (const chunk of source.chunks(signalForWrite)) {
        signalForWrite.throwIfAborted();
        if (chunk.byteLength > 1024 * 1024) throw new Error("Artifact stream chunk limit exceeded");
        size += chunk.byteLength;
        if (size > maxBytes || (source.byteLength !== undefined && size > source.byteLength)) throw new Error("Artifact size/length exceeded");
        // The producer cannot mutate a borrowed buffer while async filesystem IO consumes it.
        const owned = Buffer.from(chunk);
        sha256.update(owned); sha1.update(owned); git?.update(owned);
        await handle.writeFile(owned);
      }
      signalForWrite.throwIfAborted();
      if (source.byteLength !== undefined && size !== source.byteLength) throw new Error("Artifact length mismatch");
      const digest = sha256.digest("hex");
      if ((source.expectedSha256 && digest !== source.expectedSha256)
        || (source.expectedGitSha1 && git?.digest("hex") !== source.expectedGitSha1)) throw new Error("Artifact integrity mismatch");
      await handle.sync();
      const result = Object.freeze({ entry: Object.freeze({ path, bytes: size, sha256: digest }), sha1: sha1.digest("hex") });
      files.set(path, result);
      complete = true;
      return result;
    } finally {
      await handle.close();
      if (!complete) { await rm(target, { force: true }); names.delete(path.toLowerCase()); }
    }
  }
  function put(transfer: ArtifactTransfer, signal?: AbortSignal): Promise<StagedArtifact> {
    const operation = write(transfer, signal);
    pending.add(operation);
    void operation.then(() => pending.delete(operation), () => pending.delete(operation));
    return operation;
  }
  function graph(path: string): OnnxGraphFile {
    const { entry } = staged(path);
    return Object.freeze({ byteLength: entry.bytes, async read(offset: number, length: number) {
      staged(path);
      if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || offset < 0 || length < 0
        || offset > entry.bytes - length || length > 16 * 1024 * 1024) throw new Error("Invalid artifact read range");
      const handle = await open(pathFor(path), "r");
      try {
        const result = Buffer.alloc(length);
        let count = 0;
        while (count < length) {
          const read = await handle.read(result, count, length - count, offset + count);
          if (read.bytesRead === 0) throw new Error("Truncated staged artifact");
          count += read.bytesRead;
        }
        return result;
      } finally { await handle.close(); }
    } });
  }
  async function seal(manifest: ArtifactManifest): Promise<void> {
    writable();
    if (pending.size) throw new Error("Artifact writes still in flight");
    const supplied = createArtifactManifest(manifest.source, manifest.files);
    const actual = createArtifactManifest(manifest.source, [...files.values()].map((value) => value.entry));
    if (manifest.version !== 1 || supplied.digest !== manifest.digest || actual.digest !== supplied.digest) {
      throw new Error("Artifact manifest mismatch");
    }
    sealed = true;
    await writeFile(join(directory, ".qmd-manifest.json"), `${JSON.stringify(actual)}\n`, { flag: "wx", mode: 0o400 });
    for (const path of files.keys()) await chmod(pathFor(path), 0o400);
    for (const path of directories) await chmod(path, 0o500);
  }
  function release(): Promise<void> {
    if (releasing) return releasing;
    controller.abort(new Error("Artifact stage closing"));
    releasing = (async () => {
      await Promise.allSettled([...pending]);
      for (const path of directories) await chmod(path, 0o700);
      await rm(directory, { recursive: true, force: true });
    })();
    return releasing;
  }
  return Object.freeze({ directory, put, graph, seal, release,
    async readMetadata(path: string) {
      const reader = graph(path);
      if (reader.byteLength > 16 * 1024 * 1024) throw new Error("Artifact metadata size limit exceeded");
      return reader.read(0, reader.byteLength);
    },
  });
}
