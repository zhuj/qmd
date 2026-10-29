import { expect, test } from "vitest";
import { chmod, mkdir, mkdtemp, readFile, readdir, rm, symlink, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { createArtifactCache } from "../src/embeddings/artifact-cache.js";
import type { ArtifactCache, ArtifactCacheKey, ArtifactCacheSnapshot } from "../src/embeddings/artifact-cache.js";
import { createArtifactStage } from "../src/embeddings/artifact-files.js";
import type { ArtifactSource, ArtifactStage } from "../src/embeddings/artifact-files.js";
import { createArtifactManifest } from "../src/embeddings/identity.js";

const key: ArtifactCacheKey = { source: { kind: "huggingface", model: "fixture/encoder", revision: "a".repeat(40) }, dtype: "fp32" };
interface Fixture {
  readonly root: string; readonly directory: string; readonly cache: ArtifactCache; readonly snapshot: ArtifactCacheSnapshot;
  makeSnapshot(body: string): Promise<ArtifactCacheSnapshot>;
}
async function withFixture(run: (fixture: Fixture) => Promise<void>) {
  const root = await mkdtemp(resolve(".tmp", "artifact-cache-")), stages: ArtifactStage[] = [];
  const directory = join(root, "cache");
  try {
    const cache = createArtifactCache(directory);
    async function makeSnapshot(body: string): Promise<ArtifactCacheSnapshot> {
      const stage = await createArtifactStage(join(root, "stages")); stages.push(stage);
      const entries = [];
      for (const [path, bytes] of [["config.json", "{}"], ["onnx/model.onnx", body], ["tokenizer.json", "{}"]]) {
        if (path === undefined || bytes === undefined) throw new Error("Invalid fixture tuple");
        const source: ArtifactSource = { byteLength: Buffer.byteLength(bytes), async *chunks() { yield Buffer.from(bytes); } };
        entries.push((await stage.put({ path, source, maxBytes: 1024 })).entry);
      }
      const manifest = createArtifactManifest(key.source, entries); await stage.seal(manifest);
      return { directory: stage.directory, manifest };
    }
    await run({ root, directory, cache, snapshot: await makeSnapshot("body"), makeSnapshot });
  } finally {
    try { await Promise.all(stages.map(stage => stage.release())); }
    finally { await rm(root, { recursive: true, force: true }); }
  }
}
async function slot(directory: string) {
  const names = await readdir(join(directory, "models"));
  const name = names.find(entry => !entry.startsWith("qmd-publish-"));
  if (name === undefined) throw new Error("Missing fixture cache slot");
  return join(directory, "models", name);
}
async function collect(source: ArtifactSource) {
  const chunks: Uint8Array[] = [];
  for await (const chunk of source.chunks()) chunks.push(chunk);
  return Buffer.concat(chunks);
}
test("cache creation and offline misses do not create directories", async () => withFixture(async ({ cache, directory }) => {
  expect(await cache.find(key)).toBeUndefined();
  await expect(readdir(directory)).rejects.toMatchObject({ code: "ENOENT" });
}));
test.each(["root", "models", "slot"])("cancellation during a missing %s lookup rejects rather than returning a miss", async (missing) => withFixture(async ({ cache, directory }) => {
  if (missing !== "root") await mkdir(directory);
  if (missing === "slot") await mkdir(join(directory, "models"));
  const controller = new AbortController(), reason = new Error("cancel cache lookup");
  const pending = cache.find(key, controller.signal);
  controller.abort(reason);
  await expect(pending).rejects.toBe(reason);
  if (missing === "root") await expect(readdir(directory)).rejects.toMatchObject({ code: "ENOENT" });
  else expect(await readdir(directory)).toEqual(missing === "models" ? [] : ["models"]);
}));
test("publishes a complete snapshot, reopens it and retains the caller's stage", async () => withFixture(async ({ cache, directory, snapshot }) => {
  await cache.publish(key, snapshot);
  const found = await createArtifactCache(directory).find(key);
  if (found === undefined) throw new Error("Missing published fixture");
  expect(found.manifest).toEqual(snapshot.manifest);
  const source = await found.source("onnx/model.onnx");
  if (source === undefined) throw new Error("Missing cached payload");
  expect(source.expectedSha256).toBe(snapshot.manifest.files.find(file => file.path === "onnx/model.onnx")?.sha256);
  expect((await collect(source)).toString()).toBe("body");
  expect((await readFile(join(snapshot.directory, "onnx/model.onnx"))).toString()).toBe("body");
  expect(await found.source("unrelated.json")).toBeUndefined();
  await expect(found.source("../secret")).rejects.toThrow(/unsafe/i);
}));
test("source pin and dtype select independent slots", async () => withFixture(async ({ cache, snapshot }) => {
  await cache.publish(key, snapshot);
  expect(await cache.find({ ...key, dtype: "q8" })).toBeUndefined();
  expect(await cache.find({ ...key, source: { ...key.source, revision: "b".repeat(40) } })).toBeUndefined();
  expect((await cache.find({ ...key, source: { ...key.source, revision: "A".repeat(40) } }))?.manifest.digest).toBe(snapshot.manifest.digest);
}));
test("same manifest publication is idempotent without replacing an existing reader", async () => withFixture(async ({ cache, snapshot }) => {
  await cache.publish(key, snapshot); const original = await cache.find(key);
  if (original === undefined) throw new Error("Missing cache fixture");
  const source = await original.source("onnx/model.onnx");
  if (source === undefined) throw new Error("Missing cached source");
  await cache.publish(key, snapshot);
  expect((await collect(source)).toString()).toBe("body");
}));
test("conflicting immutable publication fails and cleans only its pending directory", async () => withFixture(async ({ cache, snapshot, makeSnapshot, directory }) => {
  await cache.publish(key, snapshot);
  await expect(cache.publish(key, await makeSnapshot("different"))).rejects.toThrow(/cache|conflict/i);
  expect((await cache.find(key))?.manifest.digest).toBe(snapshot.manifest.digest);
  expect((await readdir(join(directory, "models"))).filter(name => name.startsWith("qmd-publish-"))).toEqual([]);
}));
test("parallel complete publication supports the same and independent dtype slots", async () => withFixture(async ({ cache, snapshot, directory }) => {
  await Promise.all([cache.publish(key, snapshot), cache.publish(key, snapshot), cache.publish({ ...key, dtype: "q8" }, snapshot)]);
  expect((await readdir(join(directory, "models"))).length).toBe(2);
  expect((await cache.find(key))?.manifest.digest).toBe(snapshot.manifest.digest);
  expect((await cache.find({ ...key, dtype: "q8" }))?.manifest.digest).toBe(snapshot.manifest.digest);
}));
test.each(["digest", "version", "source", "inventory", "oversize"])("rejects invalid cached metadata: %s", async (change) => withFixture(async ({ cache, snapshot, directory }) => {
  await cache.publish(key, snapshot); const path = join(await slot(directory), ".qmd-manifest.json");
  let value: unknown = snapshot.manifest;
  switch (change) {
    case "digest": value = { ...snapshot.manifest, digest: "b".repeat(64) }; break;
    case "version": value = { ...snapshot.manifest, version: 2 }; break;
    case "source": value = { ...snapshot.manifest, source: { ...key.source, model: "other/model" } }; break;
    case "inventory": value = { ...snapshot.manifest, files: [] }; break;
    case "oversize": value = undefined; break;
    default: throw new Error("Invalid test variant");
  }
  await chmod(path, 0o600); await writeFile(path, change === "oversize" ? Buffer.alloc(1048577) : JSON.stringify(value));
  await expect(cache.find(key)).rejects.toThrow(/cache/i);
}));
test("actual staging rejects cached payload corruption, not just a hash label", async () => withFixture(async ({ cache, snapshot, directory, root }) => {
  await cache.publish(key, snapshot); const path = join(await slot(directory), "onnx/model.onnx");
  await chmod(path, 0o600); await writeFile(path, "evil");
  const found = await cache.find(key), stage = await createArtifactStage(join(root, "verification"));
  try {
    const source = await found?.source("onnx/model.onnx");
    if (source === undefined) throw new Error("Missing fixture source");
    await expect(stage.put({ path: "onnx/model.onnx", source, maxBytes: 1024 })).rejects.toThrow("Artifact integrity mismatch");
  } finally { await stage.release(); }
}));
test.each(["root", "models", "slot", "payload", "metadata"])("rejects static cached %s symlinks", async (variant) => withFixture(async ({ cache, snapshot, directory, root }) => {
  const outside = join(root, "outside"); await mkdir(outside);
  if (variant === "root") await symlink(outside, directory);
  else if (variant === "models") { await mkdir(directory); await symlink(outside, join(directory, "models")); }
  else {
    await cache.publish(key, snapshot); const directorySlot = await slot(directory);
    if (variant === "slot") { await rm(directorySlot, { recursive: true }); await symlink(outside, directorySlot); }
    else if (variant === "metadata") {
      const path = join(directorySlot, ".qmd-manifest.json"); await rm(path);
      await symlink(join(snapshot.directory, ".qmd-manifest.json"), path);
    } else {
      const path = join(directorySlot, "onnx/model.onnx"); await rm(path); await symlink(join(snapshot.directory, "onnx/model.onnx"), path);
      const found = await cache.find(key);
      await expect(found?.source("onnx/model.onnx")).rejects.toThrow(/unsafe|symlink/i);
      return;
    }
  }
  await expect(cache.find(key)).rejects.toThrow(/cache/i);
}));
test("partial copy failure removes its unpublished directory without consuming the source stage", async () => withFixture(async ({ cache, snapshot, directory }) => {
  await chmod(snapshot.directory, 0o700); await rm(join(snapshot.directory, "tokenizer.json"));
  await expect(cache.publish(key, snapshot)).rejects.toThrow(/cache publication/i);
  expect(await readdir(join(directory, "models"))).toEqual([]);
  expect((await readFile(join(snapshot.directory, "config.json"))).toString()).toBe("{}");
}));
test("cached file sources retain lookup and per-source cancellation", async () => withFixture(async ({ cache, snapshot }) => {
  await cache.publish(key, snapshot); const controller = new AbortController(), found = await cache.find(key, controller.signal);
  if (found === undefined) throw new Error("Missing cache fixture");
  controller.abort(); await expect(found.source("config.json")).rejects.toThrow();
  const fresh = await cache.find(key);
  if (fresh === undefined) throw new Error("Missing fresh cache fixture");
  await expect(fresh.source("config.json", AbortSignal.abort())).rejects.toThrow();
}));
test.each([{ bytes: Buffer.from("{") }, { bytes: Buffer.from([0xff]) }])("rejects malformed or invalid-UTF8 cached JSON", async ({ bytes }) => withFixture(async ({ cache, snapshot, directory }) => {
  await cache.publish(key, snapshot); const path = join(await slot(directory), ".qmd-manifest.json");
  await chmod(path, 0o600); await writeFile(path, bytes);
  await expect(cache.find(key)).rejects.toThrow("Cached ONNX snapshot is invalid or unavailable");
}));
test("pre-cancelled lookup/publication has no cache write side effects", async () => withFixture(async ({ cache, snapshot, directory }) => {
  await expect(cache.find(key, AbortSignal.abort())).rejects.toThrow();
  await expect(cache.publish(key, snapshot, AbortSignal.abort())).rejects.toThrow();
  await expect(readdir(directory)).rejects.toMatchObject({ code: "ENOENT" });
}));
