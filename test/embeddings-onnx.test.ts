import { expect, test } from "vitest";
import { readdir } from "node:fs/promises";
import { join } from "node:path";
import { deferred } from "./_helpers/deferred.js";
import { recipe, withFixture } from "./_helpers/onnx-capability.js";

test("creation is lazy and prepare probes dimension with CPU-only options and full identity", async () => withFixture(async f => {
  const runtime = f.port(), cap = f.capability(runtime);
  expect(runtime.observations.loads).toHaveLength(0);
  const identity = await cap.prepare();
  if (identity.kind !== "onnx-v1") throw new Error("Expected ONNX identity");
  expect([identity.descriptor.dimensions, identity.descriptor.artifactManifest.files.length]).toEqual([2, 4]);
  expect(runtime.observations.loads[0]?.options.providers).toEqual(["cpu"]);
  expect(runtime.observations.runs).toHaveLength(1);
}));

test("unsupported original target fails before artifact staging even with CPU fallback", async () => withFixture(async f => {
  const runtime = f.port(2, { platform: "win32" }), cap = f.capability(runtime, { device: "cuda", fallback: "cpu" });
  await expect(cap.prepare()).rejects.toThrow(/target|platform|unsupported/i);
  expect(runtime.observations.loads).toHaveLength(0);
  await expect(readdir(join(f.work, "sessions"))).rejects.toMatchObject({ code: "ENOENT" });
}));

test("classified accelerator initialization retries once on CPU and records sanitized provenance", async () => withFixture(async f => {
  const error = new Error("private provider detail");
  const runtime = f.port(2, { loadFailure: error, classifyLoadFailure: true });
  const cap = f.capability(runtime, { device: "cuda", fallback: "cpu" });
  const identity = await cap.prepare();
  expect(identity.kind).toBe("onnx-v1");
  expect([runtime.observations.loads.map(call => call.options.device), runtime.observations.diagnostics.length]).toEqual([["cuda", "cpu"], 1]);
  expect(JSON.stringify(runtime.observations.diagnostics)).not.toContain("private");
  expect(cap.provenance).toMatchObject({ requestedDevice: "cuda", initializedMode: "cpu", graphNodePlacement: "unknown", cpuRetryReason: "provider-initialization-failed" });
  if (cap.config.backend !== "onnx") throw new Error("Expected ONNX config");
  expect(cap.config.device).toBe("cuda");
}));

test("provider failure does not retry when fallback is disabled", async () => withFixture(async f => {
  const runtime = f.port(2, { loadFailure: new Error("private provider failure"), classifyLoadFailure: true });
  const cap = f.capability(runtime, { device: "cuda", fallback: "error" });
  const failure = await cap.prepare().catch((error: unknown) => error);
  expect(failure).toBeInstanceOf(Error);
  if (!(failure instanceof Error)) throw new Error("Expected a sanitized error");
  expect(failure.message).toMatch(/fallback.*cpu/i);
  expect(failure.message).not.toContain("private");
  expect([runtime.observations.loads.length, runtime.observations.diagnostics.length]).toEqual([1, 0]);
}));

test("ordinary load errors do not retry or expose dependency details", async () => withFixture(async f => {
  const runtime = f.port(2, { loadFailure: new Error("private model detail") });
  const cap = f.capability(runtime, { device: "cuda", fallback: "cpu" });
  const failure = await cap.prepare().catch((error: unknown) => error);
  expect(failure).toBeInstanceOf(Error);
  if (!(failure instanceof Error)) throw new Error("Expected a sanitized error");
  expect(failure.message).not.toContain("private");
  expect([runtime.observations.loads.length, runtime.observations.diagnostics.length]).toEqual([1, 0]);
}));

test("dimension mismatch fails preflight and releases session plus staged snapshot", async () => withFixture(async f => {
  const runtime = f.port(2), cap = f.capability(runtime, { dimensions: 3 });
  await expect(cap.prepare()).rejects.toThrow(/dimension/i);
  expect([runtime.observations.counts.closes, await readdir(join(f.work, "sessions"))]).toEqual([1, []]);
  const overheadPort = f.port(), noBody = f.capability(overheadPort, { recipe: { ...recipe, documentPrefix: "x".repeat(127) } });
  await expect(noBody.prepare()).rejects.toThrow(/body-token capacity/i);
  expect(overheadPort.observations.runs).toHaveLength(0); expect(overheadPort.observations.counts.closes).toBe(1);
}));

test("query and document formatting are applied once before inference", async () => withFixture(async f => {
  const runtime = f.port(), cap = f.capability(runtime);
  await cap.embed([{ kind: "query", text: "hello" }, { kind: "document", title: "t", text: "body" }]);
  expect(runtime.observations.runs.slice(1).flat()).toEqual(["q: hello", "d: t\nbody"]);
}));

test("special-token lengths produce ordered batches within padded and unpadded caps", async () => withFixture(async f => {
  const runtime = f.port(), cap = f.capability(runtime, { batchSize: 2, maxBatchTokens: 15 });
  const vectors = await cap.embed([{ kind: "query", text: "a" }, { kind: "query", text: "bb" }, { kind: "query", text: "ccc" }]);
  expect(runtime.observations.runs.slice(1)).toEqual([["q: a", "q: bb"], ["q: ccc"]]);
  expect(vectors).toHaveLength(3);
  expect(vectors[0]?.[0]).not.toBe(vectors[1]?.[0]);
  expect(vectors[1]?.[0]).not.toBe(vectors[2]?.[0]);
  expect(runtime.observations.tokenizations.some(call => call.text === "q: a" && call.special)).toBe(true);
}));

test("overlong query rejects before its user-input run without tokenizer truncation", async () => withFixture(async f => {
  const runtime = f.port(), cap = f.capability(runtime, { maxBatchTokens: 64 });
  await cap.prepare();
  const preflightRuns = runtime.observations.runs.length;
  await expect(cap.embed([{ kind: "query", text: "x".repeat(100) }])).rejects.toThrow(/token|input|limit/i);
  expect(runtime.observations.runs).toHaveLength(preflightRuns);
}));

test("aborting a queued batch preserves the caller reason and prevents its run", async () => withFixture(async f => {
  const gate = deferred<void>(), started = deferred<void>();
  const runtime = f.port(2, { userRunGate: gate, userRunStarted: started }), cap = f.capability(runtime);
  await cap.prepare();
  const first = cap.embed([{ kind: "query", text: "first" }]);
  await started.promise;
  const controller = new AbortController(), reason = new Error("queued-abort");
  const second = cap.embed([{ kind: "query", text: "second" }], controller.signal);
  controller.abort(reason);
  expect(await second.catch((error: unknown) => error)).toBe(reason);
  gate.resolve();
  expect([(await first).length, runtime.observations.runs.length]).toEqual([1, 2]);
}));

test("close drains an output lease, rejects new work, and releases owned resources once", async () => withFixture(async f => {
  const gate = deferred<void>(), releaseStarted = deferred<void>();
  const runtime = f.port(2, { userReleaseGate: gate, userReleaseStarted: releaseStarted }), cap = f.capability(runtime);
  await cap.prepare();
  const embed = cap.embed([{ kind: "query", text: "held" }]);
  await releaseStarted.promise;
  let closed = false;
  const closing = cap.close().then(() => { closed = true; });
  await Promise.resolve();
  expect(closed).toBe(false);
  await expect(cap.embed([{ kind: "query", text: "new" }])).rejects.toThrow(/clos/i);
  gate.resolve();
  await embed;
  await closing;
  expect(runtime.observations.counts.releases).toBe(2);
  expect(runtime.observations.counts.closes).toBe(1);
  await expect(readdir(join(f.work, "sessions"))).resolves.toEqual([]);
}));

test("inference failure is sanitized and does not trigger provider retry", async () => withFixture(async f => {
  const runtime = f.port(2, { userRunFailure: new Error("private inference detail") });
  const cap = f.capability(runtime, { device: "cuda", fallback: "cpu" });
  await cap.prepare();
  const failure = await cap.embed([{ kind: "query", text: "failure" }]).catch((error: unknown) => error);
  expect(failure).toBeInstanceOf(Error);
  if (!(failure instanceof Error)) throw new Error("Expected a sanitized error");
  expect(failure.message).not.toContain("private");
  expect(runtime.observations.loads).toHaveLength(1);
  expect(runtime.observations.counts.closes).toBe(0);
}));
