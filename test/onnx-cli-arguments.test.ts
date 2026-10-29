import { describe, expect, test } from "vitest";
import { readFileSync } from "node:fs";
import { createContext, runInContext } from "node:vm";
import { parseArgs } from "node:util";
import ts from "typescript";
import { resolveEmbeddingConfig, resolveSearchStagePolicy } from "../src/embeddings/config.js";
import { EmbeddingCompatibilityUnverifiedError } from "../src/store.js";
import { makeOnnxConfig } from "./_helpers/embedding-config.js";

// Keep the parser's real option declarations and complete handler bodies. Only
// filesystem/config selection, DB diagnostics and execution stages are ports.
function functions(file: string, names: readonly string[]): string {
  const source = ts.createSourceFile(file, readFileSync(new URL(file, import.meta.url), "utf8"), ts.ScriptTarget.Latest, true);
  return source.statements.filter(node => ts.isFunctionDeclaration(node)
    && node.name !== undefined && names.includes(node.name.text)).map(node => node.getText(source)).join("\n");
}
const code = ts.transpileModule([
  functions("../src/cli/qmd.ts", ["parseCLI", "parseChunkStrategy", "querySearch", "parseStructuredQuery",
    "resolveCollectionFilter", "collectionSearchFilter", "checkIndexHealth", "resolveEmbedModelForCli", "ensureModelsConfiguredForCli"]),
  functions("../src/llm.ts", ["resolveModels", "resolveEmbedModel", "resolveGenerateModel", "resolveRerankModel"]),
].join("\n"), { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None } }).outputText;

function parserHarness(args: string[]) {
  const config = { collections: {}, models: { embed: makeOnnxConfig() } };
  const embeddingConfig = resolveEmbeddingConfig(config.models.embed, { defaultGgufModel: "builtin.gguf", hostParallelism: 8 });
  const selectedStore = { db: {}, embeddingConfig, embedding: { modelLabel: "fixture/model" }, getIndexHealth: () => ({ needsEmbedding: 0, totalDocs: 1, daysStale: null }) };
  const state = { localPath: "" };
  const selection: unknown[] = [];
  const events: string[] = [];
  const stages: { query: string; expand?: boolean; skipRerank?: boolean }[] = [];
  const context = createContext({
    exports: {}, Error, parseArgs, resolveSearchStagePolicy, EmbeddingCompatibilityUnverifiedError,
    DEFAULT_EMBED_MODEL: "builtin.gguf", DEFAULT_QUERY_MODEL: "generate.gguf",
    DEFAULT_GENERATE_MODEL: "generate.gguf", DEFAULT_RERANK_MODEL: "rerank.gguf",
    process: { argv: ["node", "qmd", ...args], env: {}, stderr: { write() {} } },
    console: { error(message: string) { throw new Error(message); } }, c: {},
    findLocalConfigPath: () => state.localPath || undefined,
    setConfigSource(source?: { configPath: string }) { selection.push(source); },
    setIndexName(name: string) { selection.push(["index", name]); },
    setConfigIndexName(name: string) { selection.push(["config-index", name]); },
    getLocalDbPath: () => "/fixture/index.sqlite", storeDbPathOverride: undefined,
    closeDb() { selection.push("sync-close"); },
    loadConfig: () => config, capturedConfig: config,
    getVectorStore: async () => selectedStore,
    getIndexHealth: () => ({ needsEmbedding: 0, totalDocs: 1, daysStale: null }),
    withLLMSession: async (operation: () => Promise<unknown>) => {
      events.push("session"); try { return await operation(); } finally { events.push("release-session"); }
    },
    hybridQuery: async (_store: unknown, query: string, options: { expand?: boolean; skipRerank?: boolean }) => {
      stages.push({ query, expand: options.expand, skipRerank: options.skipRerank }); return [];
    },
    printEmptySearchResults() {},
    closeDbAsync: async () => { events.push("close"); },
  });
  runInContext(code, context);
  return { state, selection, events, stages,
    run(expression: string): unknown { return runInContext(expression, context); } };
}

describe("T14-R1 actual CLI arguments and CPU query handoff", () => {
  const cases = [
    { name: "omitted", flags: [], skipRerank: undefined },
    { name: "explicit disable", flags: ["--no-rerank"], skipRerank: true },
  ];
  test.each(cases)("actual parseArgs preserves $name intent", ({ flags, skipRerank }) => {
    const h = parserHarness(["query", "question", ...flags]);
    expect(h.run("parseCLI().opts.skipRerank")).toBe(skipRerank);
  });

  test.each(cases)("parsed $name request reaches CPU query without enabling GGUF stages", async ({ flags }) => {
    const h = parserHarness(["query", "question", ...flags]);
    await h.run("const cli = parseCLI(); querySearch(cli.query, cli.opts)");
    expect(h.stages).toEqual([{ query: "question", expand: false, skipRerank: true }]);
    expect(h.events).toEqual(["session", "release-session", "close"]);
  });

  test("explicit index selection preserves its synchronous source-selection contract", () => {
    const h = parserHarness(["query", "question", "--index", "owned-index", "--json"]);
    expect(h.run("parseCLI().opts.format")).toBe("json");
    expect(h.selection).toEqual([["index", "owned-index"], ["config-index", "owned-index"], undefined]);
  });

  test("local config discovery preserves its source, DB override and synchronous close", () => {
    const h = parserHarness(["query", "question"]);
    h.state.localPath = "/fixture/project/.qmd/index.yml";
    expect(h.run("parseCLI().query")).toBe("question");
    expect(h.selection).toEqual([{ configPath: h.state.localPath }, "sync-close"]);
    expect(h.run("storeDbPathOverride")).toBe("/fixture/index.sqlite");
  });
});
