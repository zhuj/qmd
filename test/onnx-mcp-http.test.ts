import { describe, expect, test, vi } from "vitest";
import { mkdtemp, mkdir, writeFile, readFile, rm } from "node:fs/promises";
import { existsSync, readFileSync } from "node:fs";
import { once } from "node:events";
import { availableParallelism } from "node:os";
import { join, resolve } from "node:path";
import { z } from "zod";
import { startMcpHttpServer, type HttpServerHandle } from "../src/mcp/server.js";
import { getConfigPath, loadConfig, setConfigSource } from "../src/collections.js";
import { LlamaCpp } from "../src/llm.js";
import { enableProductionMode, _resetProductionModeForTesting, getDefaultDbPath, DEFAULT_EMBED_MODEL } from "../src/store.js";
import { resolveEmbeddingConfig } from "../src/embeddings/config.js";
import type { ResolvedEmbeddingConfig } from "../src/embeddings/types.js";
import { makeOnnxConfig } from "./_helpers/embedding-config.js";

const policyError = "CPU-only ONNX policy disables query expansion and neural reranking; omit these stages or explicitly disable them";
const envKeys = ["QMD_TRUST_LOCAL_CONFIG", "QMD_TRUST_UPDATE_HOOKS", "QMD_CONFIG_DIR", "INDEX_PATH", "XDG_CACHE_HOME",
  "QMD_EMBED_MODEL", "QMD_GENERATE_MODEL", "QMD_RERANK_MODEL", "QMD_ALLOWED_ORIGINS", "QMD_ALLOWED_HOSTS"];
type Transport = "rest" | "mcp";
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
function productionMode(): boolean {
  try { getDefaultDbPath(); return true; }
  catch (error) {
    if (error instanceof Error && error.message === "Database path not set. Tests must set INDEX_PATH env var or use createStore() with explicit path. This prevents tests from accidentally writing to the global index.") return false;
    throw error;
  }
}
const textContent = z.array(z.strictObject({ type: z.literal("text"), text: z.string() })).length(1);
const manifest: unknown = JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8"));
const { version } = z.object({ version: z.string() }).parse(manifest);
const resultMeta = z.strictObject({ "io.modelcontextprotocol/serverInfo": z.strictObject({ name: z.literal("qmd"), version: z.literal(version) }) });
const completion = { resultType: "complete", _meta: { "io.modelcontextprotocol/serverInfo": { name: "qmd", version } } };
const mcpResponse = z.strictObject({ jsonrpc: z.literal("2.0"), id: z.literal(1), result: z.union([
  z.strictObject({ resultType: z.literal("complete"), _meta: resultMeta, content: textContent, structuredContent: z.strictObject({ results: z.array(z.never()) }) }),
  z.strictObject({ resultType: z.literal("complete"), _meta: resultMeta, content: textContent, isError: z.literal(true) }),
]) });
const restResponse = z.strictObject({ results: z.array(z.never()) });
interface Fixture {
  url: string;
  handle: HttpServerHandle;
  secondDbPath: string;
  calls: string[];
  errors: unknown[][];
  configPath: string;
  expected: ResolvedEmbeddingConfig;
}

async function stopOwned(handle: HttpServerHandle): Promise<void> {
  const closed = handle.httpServer.listening
    ? once(handle.httpServer, "close", { signal: AbortSignal.timeout(5000) }) : Promise.resolve();
  try { await handle.stop(); }
  finally { if (handle.httpServer.listening) handle.httpServer.close(); handle.httpServer.closeAllConnections(); await closed; }
}

async function withServer(device: "cpu" | "cuda", run: (fixture: Fixture) => Promise<void>): Promise<void> {
  const savedEnv = new Map(envKeys.map(key => [key, process.env[key]]));
  const originalPath = getConfigPath();
  const inline = originalPath === "<inline>" ? loadConfig() : undefined;
  const signals = signalNames.map(signal => ({ signal, listeners: process.listeners(signal) }));
  const root = await mkdtemp(resolve(".tmp", "onnx-mcp-http-"));
  const calls: string[] = [], errors: unknown[][] = [], failures: unknown[] = [];
  let previousSource: Parameters<typeof setConfigSource>[0];
  let previousMode: boolean | undefined;
  let sourceChanged = false;
  let restoreGuards: (() => void) | undefined;
  let restoreLogs: (() => void) | undefined;
  let handle: HttpServerHandle | undefined;
  let primary: { error: unknown } | undefined;
  const cleanup = async (action: () => void | Promise<void>) => {
    try { await action(); } catch (error) { failures.push(error); }
  };
  try {
    for (const key of envKeys) delete process.env[key];
    process.env.QMD_CONFIG_DIR = join(root, "config");
    previousSource = inline ? { config: inline } : getConfigPath() === originalPath ? { configPath: originalPath } : undefined;
    process.env.XDG_CACHE_HOME = join(root, "cache");
    previousMode = productionMode();
    const dbPath = join(root, "owned.sqlite");
    process.env.INDEX_PATH = dbPath;
    await mkdir(join(root, "config"));
    const configPath = join(root, "config", "index.yml");
    const embed = makeOnnxConfig({ source: { kind: "local", path: join(root, "missing-model") }, device });
    await writeFile(configPath, JSON.stringify({ collections: {}, models: { embed } }));
    const expected = resolveEmbeddingConfig(embed, { configPath, defaultGgufModel: DEFAULT_EMBED_MODEL, hostParallelism: availableParallelism() });
    restoreGuards = installGuards(calls);
    const logs = vi.spyOn(console, "error");
    restoreLogs = () => logs.mockRestore();
    logs.mockImplementation((...args: unknown[]) => { errors.push(args); });
    setConfigSource({ configPath });
    sourceChanged = true;
    handle = await startMcpHttpServer(0, { quiet: true, host: "127.0.0.1", dbPath });
    expect(handle.httpServer.listening).toBe(true);
    await run({ url: `http://127.0.0.1:${handle.port}`, handle, secondDbPath: join(root, "second.sqlite"), calls, errors, configPath, expected });
  } catch (error) { primary = { error }; throw error; }
  finally {
    await cleanup(async () => {
      if (handle) {
        const beforeStop = [...calls];
        await stopOwned(handle);
        expect(calls).toEqual(beforeStop);
      }
    });
    await cleanup(() => restoreGuards?.());
    await cleanup(() => restoreLogs?.());
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
    if (failures.length > 0) throw new AggregateError([...(primary ? [primary.error] : []), ...failures], "HTTP fixture cleanup failed");
  }
}

function post(fixture: Fixture, transport: Transport, extra: Record<string, unknown>): Promise<Response> {
  const args = { ...(transport === "rest" ? { searches: [{ type: "lex", query: "alpha" }] } : { query: "alpha" }), ...extra };
  const body = transport === "rest" ? args : { jsonrpc: "2.0", id: 1, method: "tools/call", params: {
    name: "query", arguments: args, _meta: {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": { name: "onnx-entry-test", version: "1.0.0" },
      "io.modelcontextprotocol/clientCapabilities": {},
    },
  } };
  return fetch(`${fixture.url}/${transport === "rest" ? "query" : "mcp"}`, {
    method: "POST", headers: {
      "Content-Type": "application/json", Accept: "application/json, text/event-stream", Connection: "close",
      "MCP-Protocol-Version": "2026-07-28", "Mcp-Method": "tools/call", "Mcp-Name": "query",
    }, body: JSON.stringify(body), signal: AbortSignal.timeout(5000),
  });
}
async function expectSuccess(fixture: Fixture, transport: Transport, response: Response): Promise<void> {
  expect(fixture.calls).toEqual([]);
  expect(response.status).toBe(200);
  const raw: unknown = await response.json();
  if (transport === "rest") expect(restResponse.parse(raw)).toEqual({ results: [] });
  else expect(mcpResponse.parse(raw)).toEqual({ jsonrpc: "2.0", id: 1, result: {
    ...completion, content: [{ type: "text", text: 'No results found for "alpha"' }], structuredContent: { results: [] },
  } });
  expect(fixture.errors).toEqual([]);
}

describe("T15 real loopback entry contracts without inference", () => {
  test("T15-AUD-01 inline-source restart fails closed before a second DB or transport", async () => {
    await withServer("cpu", async fixture => {
      expect(getConfigPath()).toBe("<inline>");
      await stopOwned(fixture.handle);
      expect(getConfigPath()).toBe("<inline>");
      let second: HttpServerHandle | undefined;
      let failure: unknown;
      try {
        try {
          second = await startMcpHttpServer(0, { quiet: true, host: "127.0.0.1", dbPath: fixture.secondDbPath });
        } catch (error) { failure = error; }
        expect(fixture.calls).toEqual([]);
        expect(existsSync(fixture.secondDbPath)).toBe(false);
        expect(second === undefined).toBe(true);
        expect(z.instanceof(Error).parse(failure).message).toBe(
          "MCP entry startup requires a file-backed configuration source; select its config file again before restarting",
        );
      } finally { if (second) await stopOwned(second); }
    });
  });
  const transports: Transport[] = ["rest", "mcp"];
  const cases = [{ label: "omitted", rerank: undefined }, { label: "false", rerank: false }, { label: "true", rerank: true }];
  for (const transport of transports) {
    test.each(cases)(`${transport} CPU rerank=$label`, async ({ rerank }) => {
      await withServer("cpu", async fixture => {
        const response = await post(fixture, transport, rerank === undefined ? {} : { rerank });
        expect(fixture.calls).toEqual([]);
        if (rerank !== true) await expectSuccess(fixture, transport, response);
        else if (transport === "rest") {
          expect(response.status).toBe(500);
          expect(await response.text()).toBe("Internal Server Error");
          expect(fixture.errors).toHaveLength(1);
          const logged = z.tuple([z.literal("HTTP handler error:"), z.instanceof(Error)]).parse(fixture.errors[0]);
          expect(logged[1].message).toBe(policyError);
        } else {
          expect(response.status).toBe(200);
          const raw: unknown = await response.json();
          expect(mcpResponse.parse(raw)).toEqual({ jsonrpc: "2.0", id: 1, result: {
            ...completion, content: [{ type: "text", text: policyError }], isError: true,
          } });
          expect(fixture.errors).toEqual([]);
        }
      });
    });
    test(`${transport} remote model/device/acquisition fields cannot change configured producer`, async () => {
      await withServer("cpu", async fixture => {
        const before = await readFile(fixture.configPath, "utf8");
        const response = await post(fixture, transport, { model: "unapproved.gguf", device: "cuda", allowDownload: true,
          source: { kind: "local", path: "/unapproved/model" }, models: { embed: makeOnnxConfig({ device: "cuda", allowDownload: true }) } });
        await expectSuccess(fixture, transport, response);
        const actual = resolveEmbeddingConfig(loadConfig().models?.embed, {
          configPath: fixture.configPath, defaultGgufModel: DEFAULT_EMBED_MODEL, hostParallelism: availableParallelism(),
        });
        expect(actual).toEqual(fixture.expected);
        expect(await readFile(fixture.configPath, "utf8")).toBe(before);
      });
    });
  }
  test("accelerated MCP defaults retain expansion independently of CPU policy", async () => {
    await withServer("cuda", async fixture => {
      const response = await post(fixture, "mcp", {});
      expect(fixture.calls).toEqual(["expandQuery"]);
      expect(response.status).toBe(200);
      const raw: unknown = await response.json();
      expect(mcpResponse.parse(raw)).toEqual({ jsonrpc: "2.0", id: 1, result: {
        ...completion, content: [{ type: "text", text: "forbidden GGUF expandQuery" }], isError: true,
      } });
      expect(fixture.errors).toEqual([]);
    });
  });
  test("accelerated REST accepts independent explicit reranking without a CPU-policy conflict", async () => {
    await withServer("cuda", async fixture => { await expectSuccess(fixture, "rest", await post(fixture, "rest", { rerank: true })); });
  });
});
