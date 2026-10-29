import { expect, test } from "vitest";
import { createHash } from "node:crypto";
import { mkdir, mkdtemp, readFile, readdir, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { createArtifactIO } from "../src/embeddings/artifact-io.js";
import { prepareArtifactSnapshot } from "../src/embeddings/artifacts.js";
import type { ArtifactSnapshot } from "../src/embeddings/artifacts.js";
import type { ArtifactHttpTransport } from "../src/embeddings/artifact-http-policy.js";
import { createArtifactManifest } from "../src/embeddings/identity.js";
import { resolveEmbeddingConfig } from "../src/embeddings/config.js";
import type { OnnxEmbeddingConfig, ResolvedOnnxConfig } from "../src/embeddings/types.js";
import { makeOnnxConfig } from "./_helpers/embedding-config.js";

function config(overrides: Partial<OnnxEmbeddingConfig> = {}): ResolvedOnnxConfig {
  const parsed = resolveEmbeddingConfig(makeOnnxConfig(overrides), { defaultGgufModel: "legacy", hostParallelism: 8 });
  if (parsed.backend !== "onnx") throw new Error("Invalid fixture backend");
  return parsed;
}
function graph(external: Readonly<Record<string, string>> = {}): Buffer {
  const field = (number: number, bytes: Uint8Array) => Buffer.concat([Buffer.from([number * 8 + 2, bytes.length]), bytes]);
  const entries = Object.entries(external).map(([key, value]) => field(13, Buffer.concat([
    field(1, Buffer.from(key)), field(2, Buffer.from(value)),
  ])));
  const tensor = Buffer.concat([...entries, Buffer.from([112, 1])]);
  return Buffer.concat([Buffer.from([8, 10]), field(7, entries.length ? field(5, tensor) : Buffer.alloc(0))]);
}
function files(external: Readonly<Record<string, string>> = {}) {
  return new Map([
    ["config.json", Buffer.from(JSON.stringify({ model_type: "bert", hidden_size: 2, max_position_embeddings: 512 }))],
    ["tokenizer_config.json", Buffer.from('{"tokenizer_class":"BertTokenizer"}')],
    ["tokenizer.json", Buffer.from('{"model":{"type":"WordPiece","vocab":{"[UNK]":0},"unk_token":"[UNK]"}}')],
    ["onnx/model.onnx", graph(external)],
  ]);
}
interface Fixture {
  readonly root: string; readonly work: string; readonly local: string;
  readonly bodies: Map<string, Buffer>; readonly snapshots: ArtifactSnapshot[];
  readonly calls: string[]; readonly transport: ArtifactHttpTransport;
}
async function withFixture(run: (fixture: Fixture) => Promise<void>, external: Readonly<Record<string, string>> = {}) {
  const root = await mkdtemp(resolve(".tmp", "onnx-artifacts-")), snapshots: ArtifactSnapshot[] = [];
  try {
    const local = join(root, "source"), bodies = files(external), calls: string[] = [];
    for (const [path, bytes] of bodies) { await mkdir(dirname(join(local, path)), { recursive: true }); await writeFile(join(local, path), bytes); }
    const transport: ArtifactHttpTransport = async request => {
      const path = request.url.pathname.split(`/resolve/${"a".repeat(40)}/`)[1], bytes = path === undefined ? undefined : bodies.get(path);
      calls.push(`${request.method}:${path}`);
      const headers = new Headers({ "x-repo-commit": "a".repeat(40) });
      if (bytes !== undefined) {
        headers.set("content-length", String(bytes.length)); headers.set("etag", createHash("sha256").update(bytes).digest("hex"));
      }
      return { status: bytes === undefined ? 404 : 200, headers,
        async *chunks() { if (bytes !== undefined) yield bytes; }, close() {} };
    };
    await run({ root, work: join(root, "work"), local, bodies, snapshots, calls, transport });
  } finally {
    try { await Promise.all(snapshots.map(snapshot => snapshot.release())); }
    finally { await rm(root, { recursive: true, force: true }); }
  }
}
test("default offline miss performs no acquisition or staging writes", async () => withFixture(async f => {
  await expect(prepareArtifactSnapshot(config(), createArtifactIO(f.work, f.transport))).rejects.toThrow(/offline|cache/i);
  expect(f.calls).toEqual([]);
  await expect(readdir(f.work)).rejects.toMatchObject({ code: "ENOENT" });
}));
test("explicit fake acquisition publishes a sealed snapshot reusable offline with independent ownership", async () => withFixture(async f => {
  const io = createArtifactIO(f.work, f.transport), first = await prepareArtifactSnapshot(config({ allowDownload: true }), io);
  f.snapshots.push(first); expect(first.manifest.files).toHaveLength(4); expect(first.profile.hiddenSize).toBe(2);
  expect(await readFile(join(first.directory, "onnx/model.onnx"))).toEqual(f.bodies.get("onnx/model.onnx"));
  expect(f.calls).toHaveLength(8);
  const second = await prepareArtifactSnapshot(config(), io); f.snapshots.push(second);
  expect(second.directory).not.toBe(first.directory); expect(second.manifest).toEqual(first.manifest); expect(f.calls).toHaveLength(8);
  await first.release(); expect((await readFile(join(second.directory, "config.json"))).length).toBeGreaterThan(0);
}));
test("local byte changes on reopen produce a new manifest without changing the old snapshot", async () => withFixture(async f => {
  const selected = config({ source: { kind: "local", path: f.local } }), io = createArtifactIO(f.work, f.transport);
  const first = await prepareArtifactSnapshot(selected, io); f.snapshots.push(first);
  const original = await readFile(join(first.directory, "tokenizer.json"));
  await writeFile(join(f.local, "tokenizer.json"), '{"model":{"type":"WordPiece","vocab":{"[UNK]":0,"new":1}}}');
  const second = await prepareArtifactSnapshot(selected, io); f.snapshots.push(second);
  expect(second.manifest.digest).not.toBe(first.manifest.digest);
  expect(await readFile(join(first.directory, "tokenizer.json"))).toEqual(original); expect(f.calls).toEqual([]);
}));
test("missing q8 does not substitute an available fp32 graph", async () => withFixture(async f => {
  await expect(prepareArtifactSnapshot(config({ source: { kind: "local", path: f.local }, dtype: "q8" }), createArtifactIO(f.work, f.transport)))
    .rejects.toThrow(/quantized|selected|artifact/i);
  expect(f.calls).toEqual([]); expect(await readdir(join(f.work, "sessions"))).toEqual([]);
}));
test("q8 selects the exact quantized artifact even when both graph variants exist", async () => withFixture(async f => {
  f.bodies.set("onnx/model_quantized.onnx", graph());
  const snapshot = await prepareArtifactSnapshot(config({ dtype: "q8", allowDownload: true }), createArtifactIO(f.work, f.transport));
  f.snapshots.push(snapshot);
  expect(snapshot.manifest.files.some(file => file.path === "onnx/model_quantized.onnx")).toBe(true);
  expect(snapshot.manifest.files.some(file => file.path === "onnx/model.onnx")).toBe(false);
  expect(f.calls.some(call => call === "GET:onnx/model.onnx")).toBe(false);
}));
test("an incomplete cached snapshot never silently falls back to fresh acquisition", async () => withFixture(async f => {
  const io = createArtifactIO(f.work, f.transport), first = await prepareArtifactSnapshot(config({ allowDownload: true }), io);
  f.snapshots.push(first);
  const models = join(f.work, "cache/models"), name = (await readdir(models))[0];
  if (name === undefined) throw new Error("Missing fixture cache slot");
  await rm(join(models, name, "onnx/model.onnx"));
  await expect(prepareArtifactSnapshot(config({ allowDownload: true }), io)).rejects.toThrow(/missing|artifact/i);
  expect(f.calls).toHaveLength(8);
  expect(await readFile(join(first.directory, "onnx/model.onnx"))).toEqual(f.bodies.get("onnx/model.onnx"));
}));
test("pre-cancellation rejects without invoking source or stage work", async () => withFixture(async f => {
  const reason = new Error("cancel before artifact preparation");
  await expect(prepareArtifactSnapshot(config(), createArtifactIO(f.work, f.transport), AbortSignal.abort(reason))).rejects.toBe(reason);
  expect(f.calls).toEqual([]); await expect(readdir(f.work)).rejects.toMatchObject({ code: "ENOENT" });
}));
test("unknown export fails before loading weights and cleans owned staging", async () => withFixture(async f => {
  f.bodies.set("config.json", Buffer.from('{"model_type":"gpt2"}'));
  await expect(prepareArtifactSnapshot(config({ allowDownload: true }), createArtifactIO(f.work, f.transport))).rejects.toThrow(/profile|export/i);
  expect(f.calls.some(call => call.includes(".onnx"))).toBe(false); expect(await readdir(join(f.work, "sessions"))).toEqual([]);
}));
test.each([{ path: "config.json", bytes: Buffer.from('{"private_marker":"fixture",') },
  { path: "tokenizer.json", bytes: Buffer.from([0xff]) }])("malformed metadata fails without exposing its contents", async ({ path, bytes }) => withFixture(async f => {
  f.bodies.set(path, bytes);
  await expect(prepareArtifactSnapshot(config({ allowDownload: true }), createArtifactIO(f.work, f.transport)))
    .rejects.toThrow("Invalid or oversized ONNX artifact JSON metadata");
  expect(await readdir(join(f.work, "sessions"))).toEqual([]);
}));
test.each([{ path: "config.json", limit: 16 * 1024 ** 2 }, { path: "onnx/model.onnx", limit: 8 * 1024 ** 3 }])(
  "declared artifact limits reject before requesting oversized bodies", async ({ path, limit }) => withFixture(async f => {
    const transport: ArtifactHttpTransport = async request => {
      const response = await f.transport(request);
      if (request.method !== "HEAD" || !request.url.pathname.endsWith(`/${path}`)) return response;
      const headers = new Headers(response.headers); headers.set("content-length", String(limit + 1));
      return { ...response, headers };
    };
    await expect(prepareArtifactSnapshot(config({ allowDownload: true }), createArtifactIO(f.work, transport))).rejects.toThrow(/size|artifact/i);
    expect(f.calls).not.toContain(`GET:${path}`); expect(await readdir(join(f.work, "sessions"))).toEqual([]);
  }),
);
test("external data is copied with whole-file checksum and declared range validation", async () => withFixture(async f => {
  const bytes = Buffer.from("0123"), path = join(f.local, "onnx/weights.bin"); await writeFile(path, bytes);
  const snapshot = await prepareArtifactSnapshot(config({ source: { kind: "local", path: f.local } }), createArtifactIO(f.work, f.transport));
  f.snapshots.push(snapshot); expect(snapshot.manifest.files).toHaveLength(5);
  expect(await readFile(join(snapshot.directory, "onnx/weights.bin"))).toEqual(bytes);
}, { location: "weights.bin", offset: "1", length: "2", checksum: createHash("sha1").update("0123").digest("hex") }));
const externalFailures: readonly { readonly reference: Readonly<Record<string, string>>; readonly payload: boolean }[] = [
  { reference: { location: "missing.bin" }, payload: false },
  { reference: { location: "weights.bin", offset: "1", length: "4" }, payload: true },
  { reference: { location: "weights.bin", checksum: "f".repeat(40) }, payload: true },
  { reference: { location: "../escape.bin" }, payload: false },
];
test.each(externalFailures)("rejects missing, inconsistent or unsafe external data", async ({ reference, payload }) => withFixture(async f => {
  if (payload) await writeFile(join(f.local, "onnx/weights.bin"), "0123");
  await expect(prepareArtifactSnapshot(config({ source: { kind: "local", path: f.local } }), createArtifactIO(f.work, f.transport)))
    .rejects.toThrow(/external|closure|artifact/i);
  expect(await readdir(join(f.work, "sessions"))).toEqual([]);
}, reference));
test("cache origin mismatch is rejected before reads or publication", async () => withFixture(async f => {
  const io = createArtifactIO(f.work, f.transport), selected = config();
  const manifest = createArtifactManifest({ kind: "huggingface", model: "other/model", revision: "b".repeat(40) },
    [{ path: "config.json", bytes: 2, sha256: "a".repeat(64) }]);
  await expect(prepareArtifactSnapshot(selected, { ...io, cached: async () => ({ manifest, source: async () => { throw new Error("Wrong source read"); } }) }))
    .rejects.toThrow(/source|pinned/i);
  expect(f.calls).toEqual([]); await expect(readdir(f.work)).rejects.toMatchObject({ code: "ENOENT" });
}));
test("cancellation of partial fake acquisition preserves its reason and cleans unpublished staging", async () => withFixture(async f => {
  const controller = new AbortController(), reason = new Error("stop artifact fixture");
  const transport: ArtifactHttpTransport = async request => {
    const response = await f.transport(request);
    if (request.method !== "GET" || !request.url.pathname.endsWith(".onnx")) return response;
    return { ...response, async *chunks() { for await (const bytes of response.chunks()) yield bytes; controller.abort(reason); } };
  };
  await expect(prepareArtifactSnapshot(config({ allowDownload: true }), createArtifactIO(f.work, transport), controller.signal)).rejects.toBe(reason);
  expect(await readdir(join(f.work, "sessions"))).toEqual([]);
  await expect(readdir(join(f.work, "cache"))).rejects.toMatchObject({ code: "ENOENT" });
}));
test.each(["synchronous", "asynchronous"])("returned cleanup redacts %s failures and is idempotent", async mode => withFixture(async f => {
  const base = createArtifactIO(f.work, f.transport), cleanups: Promise<void>[] = [];
  const other = await prepareArtifactSnapshot(config({ allowDownload: true }), base); f.snapshots.push(other);
  const detail = Object.assign(new Error("private cleanup fixture detail", { cause: new Error("private fixture cause") }), { path: "private fixture path" });
  let attempts = 0;
  try {
    const snapshot = await prepareArtifactSnapshot(config({ source: { kind: "local", path: f.local } }), { ...base,
      stage: async () => {
        const owned = await base.stage();
        return { ...owned, release: () => {
          attempts++; const cleanup = owned.release(); cleanups.push(cleanup);
          if (mode === "synchronous") throw detail;
          return cleanup.then(() => { throw detail; });
        } };
      },
    });
    const first = snapshot.release(), second = snapshot.release();
    const outcomes = await Promise.all([first.catch((error: unknown) => error), second.catch((error: unknown) => error)]);
    expect(second).toBe(first); expect(snapshot.release()).toBe(first); expect(attempts).toBe(1);
    for (const error of outcomes) {
      if (!(error instanceof Error)) throw new Error("Expected cleanup error");
      expect(error.message).toBe("ONNX artifact snapshot cleanup failed");
      expect("cause" in error).toBe(false); expect("path" in error).toBe(false);
    }
    expect(await readFile(join(other.directory, "onnx/model.onnx"))).toEqual(f.bodies.get("onnx/model.onnx"));
    const reopened = await prepareArtifactSnapshot(config(), base); f.snapshots.push(reopened);
    expect(reopened.manifest).toEqual(other.manifest); expect(f.calls).toHaveLength(8);
  } finally { await Promise.all(cleanups); }
}));
test.each(["missing", "cancelled"])("preparation %s remains primary when owned cleanup rejects", async mode => withFixture(async f => {
  const base = createArtifactIO(f.work, f.transport), controller = new AbortController(), reason = new Error("cancel primary fixture");
  const selected = config({ source: { kind: "local", path: f.local }, dtype: mode === "missing" ? "q8" : "fp32" });
  const pending = prepareArtifactSnapshot(selected, { ...base,
    local: async directory => {
      const reader = await base.local(directory);
      return { source: async (path, signal) => {
        const source = await reader.source(path, signal);
        if (mode === "cancelled" && path.endsWith(".onnx")) controller.abort(reason);
        return source;
      } };
    },
    stage: async () => {
      const owned = await base.stage();
      return { ...owned, release: async () => { await owned.release(); throw new Error("private secondary cleanup detail"); } };
    },
  }, controller.signal);
  if (mode === "cancelled") await expect(pending).rejects.toBe(reason);
  else await expect(pending).rejects.toThrow("Missing selected ONNX artifact: onnx/model_quantized.onnx");
  expect(await readdir(join(f.work, "sessions"))).toEqual([]);
}));
