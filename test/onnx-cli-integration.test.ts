import { afterEach, beforeEach, describe, expect, test } from "vitest";
import { readFileSync } from "node:fs";
import { createContext, runInContext } from "node:vm";
import ts from "typescript";
import { approveEntryConfig } from "../src/embeddings/entry-config.js";
import { resolveEmbeddingConfig, resolveSearchStagePolicy } from "../src/embeddings/config.js";
import { parseCollectionConfig, type CollectionConfig } from "../src/collections.js";
import { decideLocalConfigGate, gatedItems, hasGatedItems } from "../src/trust.js";
import { makeOnnxConfig } from "./_helpers/embedding-config.js";
import { deferred } from "./_helpers/deferred.js";

const trustEnvKeys = ["QMD_TRUST_LOCAL_CONFIG", "QMD_TRUST_UPDATE_HOOKS"];
const savedTrustEnv = new Map<string, string | undefined>();
beforeEach(() => {
  for (const key of trustEnvKeys) { savedTrustEnv.set(key, process.env[key]); delete process.env[key]; }
});
afterEach(() => {
  for (const [key, value] of savedTrustEnv) {
    if (value === undefined) delete process.env[key]; else process.env[key] = value;
  }
});

// Execute complete, parser-selected production declarations, not a reimplementation
// or the CLI entrypoint. All I/O, stores, stages and constructors are fake ports.
function declarations(file: string, names: readonly string[]): string {
  const source = ts.createSourceFile(file, readFileSync(new URL(file, import.meta.url), "utf8"), ts.ScriptTarget.Latest, true);
  const selected = source.statements.filter(node => ts.isFunctionDeclaration(node)
    ? node.name !== undefined && names.includes(node.name.text)
    : ts.isVariableStatement(node) && node.declarationList.declarations.some(d => ts.isIdentifier(d.name) && names.includes(d.name.text)));
  return selected.map(node => node.getText(source)).join("\n");
}
const cliNames = ["store", "storeDbPathOverride", "currentIndexName", "capturedConfig", "capturedConfigPath", "capturedModels", "capturedModelDefaults",
  "freezeCliConfig", "getStore", "getDb", "getDbPath", "closeDb", "closeDbAsync", "getVectorStore", "builtinModels", "collectSensitiveSnapshot",
  "checkIndexHealth", "ensureModelsConfiguredForCli", "resolveEmbedModelForCli",
  "localConfigGated", "resolveLocalConfigTrust", "resolveModelsForRuntime", "resolveCollectionFilter", "collectionSearchFilter", "parseStructuredQuery", "vectorIndex", "vectorSearch", "querySearch"];
const code = ts.transpileModule([
  declarations("../src/cli/qmd.ts", cliNames),
  declarations("../src/llm.ts", ["resolveModels", "resolveEmbedModel", "resolveGenerateModel", "resolveRerankModel"]),
].join("\n"), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText;
const defaults = { embed: "builtin.gguf", generate: "generate.gguf", rerank: "rerank.gguf" };
const localPath = "/fixture/project/.qmd/index.yml";

function harness(config: CollectionConfig = { collections: {}, models: { embed: makeOnnxConfig() } }) {
  const events: string[] = [];
  const closeStarted = deferred<void>();
  const state = { config, path: localPath, trusted: true, interactive: false, closeFails: false, lockFails: false,
    busy: false, syncFails: false, stageFails: false, loadFails: false, mutateAfterApproval: false, closeWait: Promise.resolve() };
  const llms: { embedModel: string; generateModel: string; rerankModel: string }[] = [];
  const attached: Parameters<typeof resolveSearchStagePolicy>[0][] = [];
  const approvedEntries: ReturnType<typeof approveEntryConfig>[] = [];
  const recordedDigests: string[] = [];
  const healthModels: string[] = [];
  const producerHealth = (model: string) => {
    events.push("health"); healthModels.push(model); return { needsEmbedding: 0, totalDocs: 1, daysStale: null };
  };
  const stages: { query: string; expand?: boolean; skipRerank?: boolean }[] = [];
  const s = { db: {}, dbPath: "/fixture/index.sqlite", llm: undefined, ownsLlm: false,
    embedding: undefined, embeddingConfig: undefined, getIndexHealth: producerHealth,
    close() { events.push("sync-close"); },
    async closeAsync() {
      events.push("close"); closeStarted.resolve(); await state.closeWait;
      if (state.closeFails) throw new Error("private cleanup detail");
    },
  };
  const context = createContext({
    structuredClone, Error, Object, Date, exports: {},
    process: { env: { QMD_EMBED_MODEL: "env.gguf", QMD_GENERATE_MODEL: "env-generate.gguf", QMD_QUERY_MODEL: "wrong.gguf" },
      stdin: { get isTTY() { return state.interactive; } }, stdout: { get isTTY() { return state.interactive; } }, stderr: { write() {} } },
    console: { log() {} }, c: {}, cursor: { hide() {}, show() {} }, progress: { indeterminate() {}, clear() {} }, isTTY: false,
    DEFAULT_EMBED_MODEL: defaults.embed, DEFAULT_QUERY_MODEL: defaults.generate, DEFAULT_GENERATE_MODEL: defaults.generate, DEFAULT_RERANK_MODEL: defaults.rerank,
    availableParallelism: () => 8, resolveEmbeddingConfig, resolveSearchStagePolicy, parseCollectionConfig,
    loadConfig() { events.push("load"); if (state.loadFails) throw new Error("missing recipe field"); return parseCollectionConfig(state.config, state.path); },
    getConfigPath: () => state.path,
    createStore() { events.push("open"); return s; },
    syncConfigToDb() { events.push("sync"); if (state.syncFails) throw new Error("sync failed"); },
    decideLocalConfigGate: (options: Parameters<typeof decideLocalConfigGate>[0]) => decideLocalConfigGate({ ...options, env: {}, trustedCheck: () => state.trusted }),
    approveEntryConfig(path: string, captured: CollectionConfig, builtins: typeof defaults, interactive: boolean) {
      const approved = approveEntryConfig(path, captured, builtins, interactive, () => state.trusted);
      approvedEntries.push(approved);
      events.push("approved");
      if (state.mutateAfterApproval) state.config.models = { embed: "changed.gguf", generate: "changed-stage.gguf" };
      return approved;
    },
    configureStoreEmbeddings(target: typeof s, embedding: Parameters<typeof resolveSearchStagePolicy>[0]) {
      events.push("attach"); attached.push(embedding);
      Object.assign(target, { embeddingConfig: embedding, embedding: { config: embedding, modelLabel: "fixture/model" } });
    },
    LlamaCpp: class {
      constructor(readonly options: typeof llms[number]) { llms.push(options); }
      get embedModelName() { return this.options.embedModel; }
      get generateModelName() { return this.options.generateModel; }
      get rerankModelName() { return this.options.rerankModel; }
      async dispose() {}
    }, setDefaultLlamaCpp() {},
    gatedItems, hasGatedItems, printGatedItems() {},
    confirmOnTty: async () => {
      state.config.models = { embed: "changed.gguf", generate: "changed-stage.gguf" };
      state.path = "/other/index.yml";
      runInContext("process.env.QMD_GENERATE_MODEL = 'changed-env.gguf'", context);
      return true;
    },
    recordTrust(path: string, digest: string) {
      expect(path).toBe(localPath); recordedDigests.push(digest); events.push("trust-write"); state.trusted = true;
    },
    getIndexHealth: (_db: unknown, model: string) => producerHealth(model),
    withLLMSession: async (operation: () => Promise<unknown>) => {
      events.push("session"); try { return await operation(); } finally { events.push("release-session"); }
    },
    hybridQuery: async (_store: unknown, query: string, options: typeof stages[number]) => {
      stages.push({ query, expand: options.expand, skipRerank: options.skipRerank });
      if (state.stageFails) throw new Error("stage failed"); return [];
    },
    structuredSearch: async () => { events.push("structured"); return []; },
    vectorSearchQuery: async () => { events.push("vector"); if (state.stageFails) throw new Error("stage failed"); return []; },
    printEmptySearchResults() {}, opts: { limit: 10, format: "json" },
    embedLockPathForDb: (path: string) => path, EMBED_LOCK_BUSY_MESSAGE: "busy",
    tryAcquireEmbedLock() { events.push("lock"); if (state.lockFails) throw new Error("lock failed"); return state.busy ? null : { release() { events.push("unlock"); } }; },
    getHashesNeedingEmbedding() { events.push("legacy-count"); return 0; },
    shortModelName: (name: string) => name,
    generateEmbeddings: async (_store: unknown, options: { model?: string }) => {
      events.push("generate"); expect(options.model === undefined || options.model === "fixture/model").toBe(true);
      if (state.stageFails) throw new Error("stage failed");
      return { durationMs: 0, chunksCopied: 0, chunksEmbedded: 0, docsProcessed: 0, errors: 0 };
    },
  });
  runInContext(code, context);
  return { state, events, s, llms, attached, approvedEntries, recordedDigests, healthModels, stages, context, closeStarted,
    run(expression: string): unknown { return runInContext(expression, context); } };
}

describe("T14c actual CLI bodies with hermetic boundary ports", () => {
  test("malformed config propagates before DB opening even with a 'missing' message", () => {
    const h = harness(); h.state.loadFails = true;
    expect(() => h.run("getStore() ")).toThrow("missing recipe field");
    expect(h.events).toEqual(["load"]);
  });
  test("actual structured validation rejects malformed settings before DB opening", () => {
    const h = harness({ collections: {}, models: { embed: makeOnnxConfig({ dimensions: 0 }) } });
    expect(() => h.run("getStore()")).toThrow();
    expect(h.events).toEqual(["load"]);
  });
  test("sync failure closes the just-opened bare store and preserves the failure", () => {
    const h = harness(); h.state.syncFails = true;
    expect(() => h.run("getStore() ")).toThrow("sync failed");
    expect(h.events).toEqual(["load", "open", "sync", "sync-close"]);
    expect(h.run("store")).toBeNull();
  });
  test.each(["vectorSearch('question', opts)", "querySearch('question', opts)"])("T14-R2 approved capture feeds health without rereads after caller mutation: %s", async expression => {
    const h = harness(); h.state.mutateAfterApproval = true;
    await h.run(expression);
    expect(h.attached).toHaveLength(1);
    expect(h.attached[0]?.backend).toBe("onnx");
    expect(h.attached[0]).toBe(h.approvedEntries[0]?.embedding);
    expect(h.attached[0]).toBe(h.approvedEntries[0]?.config.models?.embed);
    expect(h.attached[0]).toMatchObject({ source: makeOnnxConfig().source });
    expect(Object.isFrozen(h.attached[0])).toBe(true);
    expect(h.healthModels).toEqual(["fixture/model"]);
    expect(h.events.filter(e => e === "load")).toHaveLength(1);
    expect(h.events.indexOf("approved")).toBeLessThan(h.events.indexOf("attach"));
    expect(h.events).not.toContain("trust-write");
    expect(h.llms.at(-1)?.embedModel).not.toBe("fixture/model");
    expect(h.llms.at(-1)?.generateModel).toBe("env-generate.gguf");
    expect(h.events.slice(-2)).toEqual(["release-session", "close"]);
  });
  test.each(["vectorSearch('question', opts)", "querySearch('question', opts)", "vectorIndex()"])("denied ONNX closes before factory or stage dispatch: %s", async expression => {
    const h = harness(); h.state.trusted = false;
    await expect(Promise.resolve().then(() => h.run(expression))).rejects.toThrow(/approval|trust/i);
    expect(h.attached).toEqual([]);
    expect(h.events).not.toContain("session");
    expect(h.events).not.toContain("generate");
    expect(h.events.at(-1)).toBe("close");
  });
  test("legacy configured strings and existing denied fallback retain captured environment precedence", async () => {
    for (const trusted of [true, false]) {
      const h = harness({ collections: {}, models: { embed: "custom.gguf", generate: "custom-generation.gguf", rerank: "custom-rerank.gguf" } });
      h.state.trusted = trusted;
      await h.run("vectorSearch('question', opts)");
      expect(h.llms.at(-1)).toEqual(trusted
        ? { embedModel: "custom.gguf", generateModel: "custom-generation.gguf", rerankModel: "custom-rerank.gguf" }
        : { embedModel: "env.gguf", generateModel: "env-generate.gguf", rerankModel: defaults.rerank });
      expect(h.s.ownsLlm).toBe(true);
    }
  });
  test("interactive approval uses the frozen pre-prompt config and stage selection", async () => {
    const h = harness(); h.state.trusted = false; h.state.interactive = true;
    await h.run("vectorSearch('question', opts)");
    expect(h.approvedEntries[0]?.configPath).toBe(localPath);
    expect(h.attached[0]?.backend).toBe("onnx");
    expect(h.llms.at(-1)?.generateModel).toBe("env-generate.gguf");
    expect(h.events.filter(e => e === "load")).toHaveLength(1);
    expect(h.events.filter(e => e === "trust-write")).toHaveLength(1);
    expect(h.recordedDigests).toEqual([h.approvedEntries[0]?.digest]);
  });
  test("runtime GGUF resolver never turns a denied structured model into a fallback", () => {
    const h = harness(); h.state.trusted = false;
    expect(() => h.run("resolveModelsForRuntime()")).toThrow(/approval|trust/i);
    expect(h.events).toEqual(["load"]);
  });
  test("ONNX delegates pending selection to identity-aware T11 instead of returning on a string counter", async () => {
    const h = harness(); await h.run("vectorIndex()");
    expect(h.events).toContain("generate");
    expect(h.events).not.toContain("legacy-count");
    expect(h.events.slice(-2)).toEqual(["close", "unlock"]);
  });
  test("CPU defaults disable stages while accelerated defaults retain them", async () => {
    const devices: ("cpu" | "cuda")[] = ["cpu", "cuda"];
    for (const device of devices) {
      const h = harness({ collections: {}, models: { embed: makeOnnxConfig({ device }) } });
      await h.run("querySearch('question', opts)");
      expect(h.stages).toEqual([{ query: "question", expand: device !== "cpu", skipRerank: device === "cpu" }]);
    }
  });
  test.each(["querySearch('expand: question', opts)", "querySearch('question', { ...opts, skipRerank: false })"])("CPU explicit conflicts reject before session or stage: %s", async expression => {
    const h = harness(); await expect(Promise.resolve().then(() => h.run(expression))).rejects.toThrow(/CPU-only ONNX/);
    expect(h.stages).toEqual([]); expect(h.events).not.toContain("session");
    expect(h.events.at(-1)).toBe("close");
  });
  test("accelerated explicit expansion preserves intent and strips the directive from stage input", async () => {
    const h = harness({ collections: {}, models: { embed: makeOnnxConfig({ device: "cuda" }) } });
    await h.run("querySearch('expand: question', opts)");
    expect(h.stages).toEqual([{ query: "question", expand: true, skipRerank: false }]);
  });
  test.each(["vectorSearch('question', opts)", "querySearch('question', opts)", "vectorIndex()"])("cleanup failure surfaces only without a primary error: %s", async expression => {
    for (const primary of [false, true]) {
      const h = harness(); h.state.closeFails = true; h.state.stageFails = primary;
      await expect(Promise.resolve().then(() => h.run(expression))).rejects.toThrow(primary ? "stage failed" : "Store resource cleanup failed");
      expect(h.events).toContain("close"); expect(h.run("store")).toBe(h.s);
      if (expression === "vectorIndex()") expect(h.events.at(-1)).toBe("unlock");
    }
  });
  test("lock acquisition failure and busy return both await closure", async () => {
    for (const busy of [false, true]) {
      const h = harness(); h.state.busy = busy; h.state.lockFails = !busy;
      const result = Promise.resolve().then(() => h.run("vectorIndex()"));
      if (busy) await result; else await expect(result).rejects.toThrow("lock failed");
      expect(h.events.at(-1)).toBe("close"); expect(h.events).not.toContain("generate");
    }
  });
  test("embed holds its lock and promise until asynchronous cleanup settles", async () => {
    const h = harness(), releaseClose = deferred<void>();
    h.state.closeWait = releaseClose.promise;
    const result = Promise.resolve().then(() => h.run("vectorIndex()"));
    await h.closeStarted.promise;
    expect(h.events).not.toContain("unlock");
    expect(h.run("store")).toBe(h.s);
    releaseClose.resolve(); await result;
    expect(h.events.at(-1)).toBe("unlock");
    expect(h.run("store")).toBeNull();
  });
  test("sync lexical closure clears capture by store ownership, not the current config source", () => {
    const h = harness(); h.run("getStore()"); h.state.path = "/other/index.yml"; h.run("closeDb()");
    expect(h.run("capturedConfig")).toBeUndefined(); expect(h.run("capturedConfigPath")).toBeUndefined();
    expect(h.events.at(-1)).toBe("sync-close");
  });
});
