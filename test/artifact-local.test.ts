import { expect, test } from "vitest";
import { mkdir, mkdtemp, readFile, rename, rm, symlink, utimes, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { createArtifactFileReader } from "../src/embeddings/artifact-local.js";

async function withDirectory(run: (directory: string) => Promise<void>) {
  const root = await mkdtemp(resolve(".tmp", "artifact-local-"));
  try { await run(root); } finally { await rm(root, { recursive: true, force: true }); }
}
async function collect(chunks: AsyncIterable<Uint8Array>) {
  const result: Uint8Array[] = [];
  for await (const chunk of chunks) result.push(chunk);
  return Buffer.concat(result);
}
test("reads stable local bytes in bounded chunks without changing source files", async () => withDirectory(async (root) => {
  const bytes = Buffer.alloc(140000, 0x61); await writeFile(join(root, "model.onnx"), bytes);
  const reader = await createArtifactFileReader(root), source = await reader.source("model.onnx");
  if (source === undefined) throw new Error("Missing fixture source");
  expect(source.byteLength).toBe(bytes.length);
  const chunks: Uint8Array[] = [];
  for await (const chunk of source.chunks()) { expect(chunk.length).toBeLessThanOrEqual(65536); chunks.push(chunk); }
  expect(chunks.length).toBe(3);
  expect(Buffer.concat(chunks)).toEqual(bytes);
  expect(await readFile(join(root, "model.onnx"))).toEqual(bytes);
}));
test("returns undefined for absent descendants, not for a nonregular file", async () => withDirectory(async (root) => {
  const reader = await createArtifactFileReader(root);
  expect(await reader.source("missing/config.json")).toBeUndefined();
  await mkdir(join(root, "onnx"));
  await expect(reader.source("onnx")).rejects.toThrow(/regular|unsafe/i);
}));
test.each(["../secret", "/etc/passwd", "a/../b", "a\\b", "CON", "a./weights.bin"])("rejects unsafe relative source %s", async (path) => withDirectory(async (root) => {
  const reader = await createArtifactFileReader(root);
  await expect(reader.source(path)).rejects.toThrow(/Unsafe/i);
}));
test("rejects a root symlink", async () => withDirectory(async (root) => {
  await mkdir(join(root, "model")); await symlink(join(root, "model"), join(root, "alias"));
  await expect(createArtifactFileReader(join(root, "alias"))).rejects.toThrow(/directory|unsafe/i);
}));
test.each(["/", "/.", "/./", "//"])("rejects a root symlink spelled with directory-forcing suffix %s", async (suffix) => withDirectory(async (root) => {
  await mkdir(join(root, "model")); await symlink(join(root, "model"), join(root, "alias"));
  await expect(createArtifactFileReader(join(root, "alias") + suffix)).rejects.toThrow(/directory|unsafe/i);
}));
test("requires bare directory spellings while retaining an ordinary directory", async () => withDirectory(async (root) => {
  await writeFile(join(root, "config.json"), "{}");
  expect(await (await createArtifactFileReader(root)).source("config.json")).toBeDefined();
  for (const suffix of ["/", "/.", "/./", "//"]) {
    await expect(createArtifactFileReader(root + suffix)).rejects.toThrow(/directory|unsafe/i);
  }
}));
test("rejects file and intermediate symlinks before accessing their targets", async () => withDirectory(async (root) => {
  await mkdir(join(root, "model")); await mkdir(join(root, "outside"));
  await writeFile(join(root, "outside", "weights.bin"), "outside fixture");
  await symlink(join(root, "outside", "weights.bin"), join(root, "model", "link.bin"));
  await symlink(join(root, "outside"), join(root, "model", "linked"));
  const reader = await createArtifactFileReader(join(root, "model"));
  await expect(reader.source("link.bin")).rejects.toThrow(/symlink|unsafe/i);
  await expect(reader.source("linked/weights.bin")).rejects.toThrow(/symlink|unsafe/i);
}));
test("rejects same-length replacement between metadata and consumption", async () => withDirectory(async (root) => {
  await writeFile(join(root, "model.onnx"), "old");
  const reader = await createArtifactFileReader(root), source = await reader.source("model.onnx");
  if (source === undefined) throw new Error("Missing fixture source");
  await writeFile(join(root, "replacement"), "new"); await rename(join(root, "replacement"), join(root, "model.onnx"));
  await expect(collect(source.chunks())).rejects.toThrow(/changed/i);
  const fresh = await reader.source("model.onnx");
  if (fresh === undefined) throw new Error("Missing replacement source");
  expect((await collect(fresh.chunks())).toString()).toBe("new");
}));
test("rejects replacement of the captured root directory", async () => withDirectory(async (root) => {
  const path = join(root, "model"); await mkdir(path); await writeFile(join(path, "config.json"), "{}");
  const reader = await createArtifactFileReader(path);
  await rename(path, join(root, "retired")); await mkdir(path); await writeFile(join(path, "config.json"), "{}");
  await expect(reader.source("config.json")).rejects.toThrow(/changed|unsafe/i);
}));
test("honors metadata, captured outer and per-consumption cancellation", async () => withDirectory(async (root) => {
  await writeFile(join(root, "model.onnx"), "bytes"); const reader = await createArtifactFileReader(root);
  await expect(reader.source("model.onnx", AbortSignal.abort())).rejects.toThrow();
  const controller = new AbortController(), source = await reader.source("model.onnx", controller.signal);
  if (source === undefined) throw new Error("Missing fixture source");
  controller.abort(); await expect(collect(source.chunks())).rejects.toThrow();
  const fresh = await reader.source("model.onnx");
  if (fresh === undefined) throw new Error("Missing fresh source");
  await expect(collect(fresh.chunks(AbortSignal.abort()))).rejects.toThrow();
}));
test("cancellation between reads stops the source iterator", async () => withDirectory(async (root) => {
  await writeFile(join(root, "model.onnx"), Buffer.alloc(140000));
  const source = await (await createArtifactFileReader(root)).source("model.onnx");
  if (source === undefined) throw new Error("Missing fixture source");
  const controller = new AbortController(), iterator = source.chunks(controller.signal)[Symbol.asyncIterator]();
  expect((await iterator.next()).done).toBe(false);
  controller.abort(); await expect(iterator.next()).rejects.toThrow();
  expect((await iterator.next()).done).toBe(true);
}));
test("ordinary in-place modification while streaming fails before publication can finish", async () => withDirectory(async (root) => {
  const path = join(root, "model.onnx"); await writeFile(path, Buffer.alloc(140000, 0x61));
  const source = await (await createArtifactFileReader(root)).source("model.onnx");
  if (source === undefined) throw new Error("Missing fixture source");
  const iterator = source.chunks()[Symbol.asyncIterator](); await iterator.next();
  await writeFile(path, Buffer.alloc(140000, 0x62)); await utimes(path, new Date(2000, 0), new Date(2000, 0));
  const finish = async () => { while (!(await iterator.next()).done) {} };
  await expect(finish()).rejects.toThrow(/changed/i);
}));
