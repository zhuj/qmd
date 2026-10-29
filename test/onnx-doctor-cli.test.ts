import { describe, expect, test } from "vitest";
import { readFileSync } from "node:fs";
import { createContext, runInContext } from "node:vm";
import ts from "typescript";
import { resolveEmbeddingConfig } from "../src/embeddings/config.js";
import type { StoreEmbeddingDiagnostics } from "../src/store.js";
import { makeOnnxConfig } from "./_helpers/embedding-config.js";

const source = ts.createSourceFile("qmd.ts", readFileSync(new URL("../src/cli/qmd.ts", import.meta.url), "utf8"), ts.ScriptTarget.Latest, true);
function compile(name: "showDoctor" | "checkIndexHealth" | "checkDoctorIndexConfig"): string {
  const declaration = source.statements.find(node => ts.isFunctionDeclaration(node) && node.name?.text === name);
  if (!declaration) throw new Error(`Missing production declaration: ${name}`);
  return ts.transpileModule(declaration.getText(source), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
  }).outputText;
}
interface Settings {
  device?: "cpu" | "cuda";
  unbound?: boolean;
  diagnostic?: StoreEmbeddingDiagnostics;
  approvalError?: Error;
  inspectionError?: Error;
  cleanupError?: Error;
  invalidConfig?: boolean;
  storeError?: Error;
}
function fixture(settings: Settings = {}) {
  const events: string[] = [], forbidden: string[] = [], output: string[] = [];
  const checks: { label: string; ok: boolean; details: string }[] = [];
  const inspected: { store: unknown; verify: boolean }[] = [];
  const embedding = resolveEmbeddingConfig(makeOnnxConfig({ device: settings.device ?? "cpu",
    source: { kind: "local", path: "/sensitive/model/weights" } }), { defaultGgufModel: "unused", hostParallelism: 8 });
  const bare = { db: { prepare: () => ({ get: () => ({ version: "fixture" }) }) } };
  const bound = { ...bare, embeddingConfig: embedding };
  const diagnostic: StoreEmbeddingDiagnostics = settings.diagnostic ?? { backend: "onnx", identity: undefined,
    compatible: false, reason: "Compatibility unverified; producer identity has not been established",
    requestedDevice: settings.unbound ? undefined : settings.device ?? "cpu", graphNodePlacement: "unknown", sampleVerification: "not-run" };
  const diagMsg = (m: string) => m;
  const ctxGetConfigPath = () => "/fixture/index.yml";
  let configReads = 0;
  const fullConfig = settings.unbound
    ? { collections: {} }
    : { collections: {}, models: { embed: embedding } };
  const ctxLoadConfig = () => { configReads += 1; if (settings.invalidConfig) throw new Error("invalid yaml"); return fullConfig; };
  const storeConfigs: unknown[] = [];
  const banned = ["resolveModelsForCli", "getEmbeddingFingerprint", "checkModelCache", "runDoctorDeviceChecks",
    "getDefaultLlamaCpp", "getDeviceInfo", "maybeAdoptLegacyEmbeddingFingerprint", "getHashesNeedingEmbedding",
    "checkEmbeddingVectorSamples", "checkEnvironmentOverrides", "checkModelDefaults"];
  const ports = Object.fromEntries(banned.map(name => [name, () => {
    forbidden.push(name); throw new Error(`forbidden ${name}`);
  }]));
  const context = createContext({
    ...ports, Error, exports: {}, capturedConfig: { collections: {}, models: settings.unbound ? undefined : { embed: embedding } },
    getStore(config: unknown) { storeConfigs.push(config); if (settings.storeError) throw settings.storeError; events.push("bare"); return bare; },
    getVectorStore: async () => {
      events.push("approve"); if (settings.approvalError) throw settings.approvalError; return bound;
    },
    inspectStoreEmbeddings: async (store: unknown, verify: boolean) => {
      events.push("inspect"); inspected.push({ store, verify });
      if (settings.inspectionError) throw settings.inspectionError; return diagnostic;
    },
    closeDbAsync: async () => { events.push("close"); if (settings.cleanupError) throw settings.cleanupError; },
    closeDb: () => { forbidden.push("sync-close"); throw new Error("forbidden sync close"); },
    loadConfig: ctxLoadConfig,
    getConfigPath: ctxGetConfigPath,
    sanitizeDiagnosticMessage: diagMsg,
    normalizedDoctorNextSteps: (steps: string[]) => steps,
    formatCount: (n: number) => String(n),
    readPackageJson: () => ({ dependencies: {} }), getDbPath: () => "/fixture/index.sqlite", isBun: false,
    console: { log: (...parts: unknown[]) => { output.push(parts.map(String).join(" ")); } },
    c: { bold: "", reset: "", dim: "", yellow: "", green: "" },
    doctorCheck(label: string, ok: boolean, details: string) { checks.push({ label, ok, details }); output.push(`${label}: ${details}`); },
  });
  return { bare, bound, events, forbidden, inspected, checks, fullConfig, storeConfigs,
    text: () => output.join("\n"), get configReads() { return configReads; },
    async run(): Promise<void> {
      const entry: unknown = runInContext(`${compile("showDoctor")}\n${compile("checkDoctorIndexConfig")}\nshowDoctor`, context);
      if (typeof entry !== "function") throw new Error("Doctor entry is not callable");
      const pending: unknown = entry();
      if (pending === null || typeof pending !== "object" || !("then" in pending) || typeof pending.then !== "function") {
        throw new Error("Doctor entry did not return a promise");
      }
      await Promise.resolve(pending);
    } };
}

describe("T16 actual ONNX doctor dispatch without probes", () => {
  test("an unbound stored new namespace is inventory, never permission to select default GGUF", async () => {
    const f = fixture({ unbound: true });
    await f.run();
    expect(f.forbidden).toEqual([]);
    expect(f.events).toEqual(["bare", "inspect", "close"]);
    expect(f.inspected).toEqual([{ store: f.bare, verify: false }]);
    expect(f.text()).toContain("requested device: not selected");
    expect(f.text()).not.toContain("--force");
  });
  test("CPU metadata default approves, inspects false, reports unverified and closes", async () => {
    const f = fixture();
    await f.run();
    expect(f.forbidden).toEqual([]);
    expect(f.events).toEqual(["bare", "approve", "inspect", "close"]);
    expect(f.inspected).toEqual([{ store: f.bound, verify: false }]);
    expect(f.text()).toContain("requested device: cpu");
    expect(f.text()).toContain("not initialized");
    expect(f.text()).toContain("unknown");
    expect(f.text().toLowerCase()).toContain("unverified");
    expect(f.checks.some(check => !check.ok && /unverified/i.test(check.details))).toBe(true);
    expect(f.text()).not.toContain("/sensitive");
    expect(f.text()).not.toContain("--force");
  });
  test("requested CUDA differs from live CPU initialization and retry, without execution claims", async () => {
    const f = fixture({ device: "cuda", diagnostic: { backend: "onnx", identity: "onnx-v1:fixture",
      compatible: true, reason: "Prepared identity admitted", requestedDevice: "cuda", initializedMode: "cpu",
      cpuRetryReason: "provider-initialization-failed", graphNodePlacement: "unknown", sampleVerification: "not-run" } });
    await f.run();
    expect(f.forbidden).toEqual([]);
    expect(f.text()).toContain("requested device: cuda");
    expect(f.text()).toContain("initialized mode: cpu");
    expect(f.text()).toContain("provider-initialization-failed");
    expect(f.text()).toContain("unknown");
    expect(f.text().toLowerCase()).not.toMatch(/verified gpu|cuda execution/);
  });
  test("denied approval never inspects and still awaits cleanup", async () => {
    const primary = new Error("approval denied"), f = fixture({ approvalError: primary });
    await expect(f.run()).rejects.toBe(primary);
    expect(f.forbidden).toEqual([]);
    expect(f.inspected).toEqual([]);
    expect(f.events).toEqual(["bare", "approve", "close"]);
  });
  test("inspection failure survives a secondary cleanup failure", async () => {
    const primary = new Error("inspection failed"), f = fixture({ inspectionError: primary, cleanupError: new Error("cleanup failed") });
    await expect(f.run()).rejects.toBe(primary);
    expect(f.forbidden).toEqual([]);
    expect(f.events).toEqual(["bare", "approve", "inspect", "close"]);
  });
  test("cleanup-only failure is observable", async () => {
    const cleanup = new Error("cleanup failed"), f = fixture({ cleanupError: cleanup });
    await expect(f.run()).rejects.toBe(cleanup);
    expect(f.forbidden).toEqual([]);
    expect(f.events).toEqual(["bare", "approve", "inspect", "close"]);
  });
  test("invalid config yields no getStore/getVectorStore/prepare/embed and returns after preflight", async () => {
    const f = fixture({ invalidConfig: true });
    await f.run();
    expect(f.events).toEqual([]);
    expect(f.forbidden).toEqual([]);
    expect(f.text()).toContain("index config");
    expect(f.text()).toContain("invalid index.yml");
    expect(f.text()).toContain("fix the YAML");
    expect(f.storeConfigs).toEqual([]);
    expect(f.configReads).toBe(1);
  });
  test("store failure with valid config propagates, never getVectorStore/inspect", async () => {
    const primary = new Error("store constructor failed"), f = fixture({ storeError: primary });
    await expect(f.run()).rejects.toBe(primary);
    expect(f.events).toEqual([]); // getStore throws before bare/approve/inspect
    expect(f.forbidden).toEqual([]);
  });
  test("single config load and exact snapshot for valid path", async () => {
    const f = fixture();
    await f.run();
    expect(f.configReads).toBe(1);
    expect(f.storeConfigs).toHaveLength(1);
    expect(f.storeConfigs[0]).toBe(f.fullConfig);
    expect(f.forbidden).toEqual([]);
    expect(f.events).toEqual(["bare", "approve", "inspect", "close"]);
  });
});

describe("T16 producer-relative query health", () => {
  class Unverified extends Error {}
  test.each([true, false])("only the explicit unverified class permits query continuation: %s", unverified => {
    const primary = unverified ? new Unverified("compatibility unverified") : new Error("compatibility unverified");
    const calls: string[] = [], output: string[] = [];
    const bound = { db: {}, getIndexHealth() { calls.push("bound"); throw primary; } };
    const context = createContext({ Error, EmbeddingCompatibilityUnverifiedError: Unverified, bound,
      getIndexHealth: () => { calls.push("raw"); throw new Error("forbidden raw health"); },
      c: { dim: "", reset: "" }, process: { stderr: { write: (text: string) => { output.push(text); } } } });
    runInContext(compile("checkIndexHealth"), context);
    const run = () => runInContext("checkIndexHealth(bound.db, 'selected-producer', bound)", context);
    if (unverified) { expect(run).not.toThrow(); expect(output.join("").toLowerCase()).toContain("unverified"); }
    else {
      let failure: unknown;
      try { run(); } catch (error) { failure = error; }
      expect(failure).toBe(primary);
    }
    expect(calls).toEqual(["bound"]);
    expect(output.join("")).not.toContain("0 documents");
  });
});
