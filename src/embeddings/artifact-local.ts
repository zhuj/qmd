import type { ArtifactSource } from "./artifact-files.js";
import type { BigIntStats } from "node:fs";
import type { FileHandle } from "node:fs/promises";
import { constants } from "node:fs";
import { lstat, open, realpath } from "node:fs/promises";
import { join, parse } from "node:path";
import { validateArtifactPath } from "./artifact-path.js";

export interface ArtifactFileReader {
  source(path: string, signal?: AbortSignal): Promise<ArtifactSource | undefined>;
}
async function metadata(path: string): Promise<BigIntStats | undefined> {
  try { return await lstat(path, { bigint: true }); }
  catch (error) {
    if (error instanceof Error && "code" in error && error.code === "ENOENT") return undefined;
    throw new Error("Local embedding artifact metadata is unavailable");
  }
}
function sameFile(left: BigIntStats, right: BigIntStats): boolean {
  return left.dev === right.dev && left.ino === right.ino && left.size === right.size
    && left.mtimeNs === right.mtimeNs && left.ctimeNs === right.ctimeNs;
}
export async function createArtifactFileReader(directory: string): Promise<ArtifactFileReader> {
  // Inspect the selected entry before canonicalization: directory-forcing
  // suffixes would make lstat follow a symlink even without a race.
  if ((/[\\/]$/.test(directory) && parse(directory).root !== directory) || /(?:^|[\\/])\.[\\/]*$/.test(directory)) {
    throw new Error("Unsafe embedding artifact directory spelling; omit trailing separators and terminal dot components");
  }
  const selected = await metadata(directory);
  if (selected === undefined || !selected.isDirectory()) throw new Error("Unsafe local embedding artifact directory");
  const rootDevice = selected.dev, rootInode = selected.ino;
  const root = await realpath(directory).catch(() => { throw new Error("Local embedding artifact directory is unavailable"); });
  async function ensureRoot(): Promise<void> {
    const currentRoot = await metadata(root);
    if (currentRoot === undefined || !currentRoot.isDirectory() || currentRoot.dev !== rootDevice || currentRoot.ino !== rootInode) {
      throw new Error("Local embedding artifact directory changed or is unsafe");
    }
  }
  async function statPath(path: string): Promise<BigIntStats | undefined> {
    await ensureRoot();
    const parts = path.split("/");
    let parent = root;
    for (const part of parts.slice(0, -1)) {
      parent = join(parent, part);
      const entry = await metadata(parent);
      if (entry === undefined) return undefined;
      if (!entry.isDirectory()) throw new Error("Unsafe local embedding artifact ancestor or symlink");
    }
    const entry = await metadata(join(root, path));
    if (entry !== undefined && !entry.isFile()) throw new Error("Unsafe nonregular local embedding artifact or symlink");
    return entry;
  }
  // Captured root identity also detects replacement during canonicalization.
  await ensureRoot();
  return Object.freeze({
    async source(path: string, signal?: AbortSignal): Promise<ArtifactSource | undefined> {
      signal?.throwIfAborted();
      validateArtifactPath(path);
      const expected = await statPath(path);
      signal?.throwIfAborted();
      if (expected === undefined) return undefined;
      if (expected.size < 0n || expected.size > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("Local embedding artifact size is unsupported");
      const byteLength = Number(expected.size);
      return Object.freeze({ byteLength,
        async *chunks(streamSignal?: AbortSignal) {
          const active = signal && streamSignal ? AbortSignal.any([signal, streamSignal]) : signal ?? streamSignal;
          active?.throwIfAborted();
          let file: FileHandle | undefined;
          let failed = false;
          try {
            const before = await statPath(path);
            if (before === undefined || !sameFile(before, expected)) throw new Error("Artifact changed");
            // POSIX flags supplement ancestry/identity checks. This is not an
            // atomic openat sandbox against a hostile same-user filesystem actor.
            file = await open(join(root, path), constants.O_RDONLY | (constants.O_NOFOLLOW ?? 0) | (constants.O_NONBLOCK ?? 0));
            const opened = await file.stat({ bigint: true }), afterOpen = await statPath(path);
            if (!opened.isFile() || !sameFile(opened, expected) || afterOpen === undefined || !sameFile(afterOpen, expected)) throw new Error("Artifact changed");
            for (let offset = 0; offset < byteLength;) {
              active?.throwIfAborted();
              const buffer = Buffer.allocUnsafe(Math.min(65536, byteLength - offset));
              const { bytesRead } = await file.read(buffer, 0, buffer.length, offset);
              active?.throwIfAborted();
              if (bytesRead === 0) throw new Error("Artifact truncated");
              offset += bytesRead;
              yield buffer.subarray(0, bytesRead);
            }
            active?.throwIfAborted();
            if (!sameFile(await file.stat({ bigint: true }), expected)) throw new Error("Artifact changed");
          } catch {
            failed = true;
            active?.throwIfAborted();
            throw new Error("Local embedding artifact changed or could not be read");
          } finally {
            try { await file?.close(); }
            catch { if (!failed) throw new Error("Local embedding artifact could not be closed"); }
          }
        },
      });
    },
  });
}
