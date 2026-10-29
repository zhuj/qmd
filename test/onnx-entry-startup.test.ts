import { describe, expect, test, vi } from "vitest";
import { mkdtemp, rm, mkdir, writeFile } from "node:fs/promises";
import { existsSync, readFileSync } from "node:fs";
import { EventEmitter, once } from "node:events";
import { join, resolve } from "node:path";
import { createContext, runInContext } from "node:vm";
import ts from "typescript";
import { z } from "zod";
import { startMcpHttpServer, registerStdioEofShutdown, type HttpServerHandle, type StdioShutdownStdin } from "../src/mcp/server.js";
import { getConfigPath, loadConfig, setConfigSource } from "../src/collections.js";
import { approveEntryConfig } from "../src/embeddings/entry-config.js";
import { LlamaCpp } from "../src/llm.js";
import { enableProductionMode, _resetProductionModeForTesting, getDefaultDbPath,
  DEFAULT_EMBED_MODEL, DEFAULT_RERANK_MODEL, DEFAULT_QUERY_MODEL } from "../src/store.js";
import { makeOnnxConfig } from "./_helpers/embedding-config.js";

const envKeys = ["QMD_TRUST_LOCAL_CONFIG", "QMD_TRUST_UPDATE_HOOKS", "QMD_CONFIG_DIR", "INDEX_PATH", "XDG_CACHE_HOME"];
const signalNames: NodeJS.Signals[] = ["SIGINT", "SIGTERM"];
type Stage = "tokenize" | "detokenize" | "embed" | "embedBatch" | "expandQuery" | "rerank";
const methods: Stage[] = ["tokenize", "detokenize", "embed", "embedBatch", "expandQuery", "rerank"];
function installGuards(calls: string[]): () => void {
  const restores: (() => void)[] = [];
  const restore = () => { for (const undo of restores.reverse()) undo(); };
  try {
    for (const method of methods) {
      const spy = vi.spyOn(LlamaCpp.prototype, method);
      restores.push(() => spy.mockRestore());
      spy.mockImplementation(async (): Promise<never> => { calls.push(method); throw new Error(`forbidden GGUF ${method}`); });
    }
    return restore;
  } catch (error) { restore(); throw error; }
}

// An approval regression must fail before construction, never attach real stdin.
// Both complete production declarations retain their real loader/approval flow.
const source = ts.createSourceFile("server.ts", readFileSync(new URL("../src/mcp/server.ts", import.meta.url), "utf8"), ts.ScriptTarget.Latest, true);
function stdioProgram(names: readonly string[]): string {
  return ts.transpileModule(source.statements.filter(node => ts.isFunctionDeclaration(node)
    && node.name !== undefined && names.includes(node.name.text)).map(node => node.getText(source)).join("\n"), {
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.None },
  }).outputText;
}
const startupCode = stdioProgram(["buildApprovedStore", "startMcpServer"]);
function guardedStdio(dbPath: string, ports: string[]): unknown {
  const forbidden = (name: string): never => { ports.push(name); throw new Error(`forbidden startup ${name}`); };
  const context = createContext({
    exports: {}, getConfigPath, getDefaultDbPath, existsSync, loadConfig, approveEntryConfig, enableProductionMode,
    DEFAULT_EMBED_MODEL, DEFAULT_RERANK_MODEL, DEFAULT_QUERY_MODEL, dbPath,
    createStore: () => forbidden("createStore"), serveStdio: () => forbidden("serveStdio"),
    createInflightGate: () => forbidden("createInflightGate"), createMcpServer: () => forbidden("createMcpServer"),
    registerStdioEofShutdown: () => forbidden("registerStdioEofShutdown"),
  });
  runInContext(startupCode, context);
  return runInContext("startMcpServer({ dbPath })", context);
}

function productionMode(): boolean {
  try { getDefaultDbPath(); return true; }
  catch (error) {
    if (error instanceof Error && error.message === "Database path not set. Tests must set INDEX_PATH env var or use createStore() with explicit path. This prevents tests from accidentally writing to the global index.") return false;
    throw error;
  }
}

describe("T15 denied/invalid startup before construction", () => {
  const cases = [
    { transport: "stdio", invalid: false }, { transport: "http", invalid: false },
    { transport: "stdio", invalid: true }, { transport: "http", invalid: true },
  ];
  test.each(cases)("$transport invalid=$invalid rejects before DB/transport", async ({ transport, invalid }) => {
    const savedEnv = new Map(envKeys.map(key => [key, process.env[key]]));
    const originalPath = getConfigPath();
    const inline = originalPath === "<inline>" ? loadConfig() : undefined;
    const signals = signalNames.map(signal => ({ signal, listeners: process.listeners(signal) }));
    const stdin = ["end", "close"].map(event => ({ event, listeners: process.stdin.listeners(event) }));
    const root = await mkdtemp(resolve(".tmp", "onnx-entry-startup-"));
    const calls: string[] = [], ports: string[] = [], failures: unknown[] = [];
    let previousSource: Parameters<typeof setConfigSource>[0];
    let previousMode: boolean | undefined;
    let sourceChanged = false;
    let restoreGuards: (() => void) | undefined;
    let handle: HttpServerHandle | undefined;
    let primary: { error: unknown } | undefined;
    const cleanup = async (action: () => void | Promise<void>) => {
      try { await action(); } catch (error) { failures.push(error); }
    };
    try {
      for (const key of envKeys) delete process.env[key];
      process.env.QMD_CONFIG_DIR = join(root, "global");
      previousSource = inline ? { config: inline } : getConfigPath() === originalPath ? { configPath: originalPath } : undefined;
      process.env.XDG_CACHE_HOME = join(root, "cache");
      previousMode = productionMode();
      const dbPath = join(root, "owned.sqlite");
      process.env.INDEX_PATH = dbPath;
      if (invalid) process.env.QMD_TRUST_LOCAL_CONFIG = "1";
      await mkdir(join(root, ".qmd"));
      const configPath = join(root, ".qmd", "index.yml");
      const embed = makeOnnxConfig({ source: { kind: "local", path: join(root, "missing-model") }, ...(invalid ? { dimensions: 0 } : {}) });
      await writeFile(configPath, JSON.stringify({ collections: {}, models: { embed } }));
      restoreGuards = installGuards(calls);
      setConfigSource({ configPath });
      sourceChanged = true;
      let failure: unknown;
      try {
        if (transport === "stdio") await guardedStdio(dbPath, ports);
        else handle = await startMcpHttpServer(0, { quiet: true, host: "127.0.0.1", dbPath });
      } catch (error) { failure = error; }
      expect(calls).toEqual([]);
      expect(ports).toEqual([]);
      expect(existsSync(dbPath)).toBe(false);
      expect(handle).toBeUndefined();
      for (const { event, listeners } of stdin) expect(process.stdin.listeners(event)).toEqual(listeners);
      const message = z.instanceof(Error).parse(failure).message;
      if (invalid) {
        expect(message).toContain("dimensions");
        expect(message).toContain("Too small: expected number to be >0");
      } else {
        expect(message).toBe("Project-local configuration approval required; review and trust this configuration before using its models.");
      }
    } catch (error) { primary = { error }; throw error; }
    finally {
      await cleanup(async () => {
        if (handle) {
          const closed = once(handle.httpServer, "close", { signal: AbortSignal.timeout(5000) });
          try { await handle.stop(); }
          finally { if (handle.httpServer.listening) handle.httpServer.close(); handle.httpServer.closeAllConnections(); await closed; }
        }
        expect(calls).toEqual([]);
      });
      await cleanup(() => restoreGuards?.());
      await cleanup(() => {
        for (const { signal, listeners } of signals) {
          for (const listener of process.listeners(signal)) if (!listeners.includes(listener)) process.removeListener(signal, listener);
          expect(process.listeners(signal)).toEqual(listeners);
        }
      });
      await cleanup(() => {
        if (previousMode !== undefined) { if (previousMode) enableProductionMode(); else _resetProductionModeForTesting(); }
        if (sourceChanged) setConfigSource(previousSource);
        for (const [key, value] of savedEnv) { if (value === undefined) delete process.env[key]; else process.env[key] = value; }
      });
      await cleanup(() => rm(root, { recursive: true, force: true }));
      if (failures.length > 0) throw new AggregateError([...(primary ? [primary.error] : []), ...failures], "startup fixture cleanup failed");
    }
  });
});

describe("T15-AUD-02 partial stdio startup ownership", () => {
  test.each([false, true])("registration failure closes handle before store; handle close fails=%s", async closeFails => {
    const events: string[] = [];
    const primary = new Error("EOF registration failed");
    const handle = { async close() {
      events.push("handle-close");
      if (closeFails) throw new Error("secondary handle close failed");
    } };
    const ownedStore = { async close() { events.push("store-close"); } };
    const context = createContext({
      exports: {}, enableProductionMode() {},
      buildApprovedStore: async () => ownedStore,
      createInflightGate: () => ({ waitForIdle: async () => true }),
      serveStdio: () => handle,
      createMcpServer: () => { throw new Error("unexpected transport callback"); },
      registerStdioEofShutdown: () => { throw primary; },
    });
    runInContext(stdioProgram(["startMcpServer"]), context);
    const result: unknown = runInContext("startMcpServer()", context);
    await expect(Promise.resolve(result)).rejects.toBe(primary);
    expect(events).toEqual(["handle-close", "store-close"]);
  });

  test.each([false, true])("failed EOF listener registration rolls back independently; end off fails=%s", async offFails => {
    const emitter = new EventEmitter();
    const events: string[] = [], removals: string[] = [];
    const foreign = () => { events.push("foreign-listener"); };
    const primary = new Error("close listener registration failed");
    emitter.on("end", foreign);
    emitter.on("newListener", (event: string | symbol) => { if (event === "close") throw primary; });
    const stdin: StdioShutdownStdin = {
      once(event, listener) { return emitter.once(event, listener); },
      off(event, listener) {
        removals.push(event);
        emitter.off(event, listener);
        if (offFails && event === "end") throw new Error("secondary end removal failed");
      },
    };
    try {
      let failure: unknown;
      try {
        registerStdioEofShutdown({
          stdin, stderr: { write() { events.push("stderr-write"); }, on() {} },
          closeServer: async () => { events.push("server-close"); },
          closeStore: async () => { events.push("store-close"); },
          setExitCode: () => { events.push("exit-code"); },
          getExitCode: () => { events.push("get-exit-code"); return undefined; },
        });
      } catch (error) { failure = error; }
      await Promise.resolve();
      expect(failure).toBe(primary);
      expect(events).toEqual([]);
      expect(emitter.listeners("end")).toEqual([foreign]);
      expect(emitter.listeners("close")).toEqual([]);
      expect(removals).toEqual(["end", "close"]);
    } finally { emitter.removeAllListeners(); }
  });
});
