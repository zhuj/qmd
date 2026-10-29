import { describe, expect, test, beforeEach, afterEach } from "vitest";
import { approveEntryConfig } from "../src/embeddings/entry-config.js";
import type { CollectionConfig } from "../src/collections.js";
import type { BuiltinModels } from "../src/trust.js";
import { isLocalConfigPath } from "../src/trust.js";
import { makeOnnxConfig } from "./_helpers/embedding-config.js";
import type { OnnxEmbeddingConfig } from "../src/embeddings/types.js";
import { resolveEmbedModelForCli, resolveGenerateModelForCli, resolveRerankModelForCli } from "../src/cli/qmd.js";
import { setConfigSource, loadConfig } from "../src/collections.js";
import { dirname, resolve } from "path";

const fixtureConfigPath = "/fixture/project/.qmd/index.yml";
const globalConfigPath = "/fixture/home/.config/qmd/index.yml";
const builtins: BuiltinModels = { embed: "builtin.gguf", rerank: "builtin.gguf", generate: "builtin.gguf" };

const ENV_KEYS: readonly ("QMD_TRUST_LOCAL_CONFIG" | "QMD_TRUST_UPDATE_HOOKS" | "QMD_EMBED_MODEL")[] = ["QMD_TRUST_LOCAL_CONFIG", "QMD_TRUST_UPDATE_HOOKS", "QMD_EMBED_MODEL"];

function mutableLocalSource(path: string): { kind: "local"; path: string } {
  return { kind: "local", path };
}

describe("T14b approveEntryConfig policy test", () => {
  let saved: Record<string, string | undefined> = {};

  beforeEach(() => {
    saved = {};
    for (const k of ENV_KEYS) saved[k] = process.env[k];
    for (const k of ENV_KEYS) delete process.env[k];
  });

  afterEach(() => {
    for (const k of ENV_KEYS) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k];
    }
  });

  const cases = [
    {
      name: "approved HF descriptor frozen digest exact defaults freeze",
      run() {
        const baseRecipe = makeOnnxConfig().recipe;
        const embedConfig: OnnxEmbeddingConfig = { backend: "onnx", source: { kind: "huggingface", model: "fixture/model", revision: "a".repeat(40) }, recipe: baseRecipe, device: "cpu", fallback: "error", dtype: "fp32", allowDownload: false };
        const config: CollectionConfig = { collections: { docs: { path: "./docs", pattern: "**/*.md" } }, models: { embed: embedConfig } };
        const calls: string[] = [];
        const trusted = (p: string, d: string) => { calls.push(p); calls.push(d); return true; };
        const res = approveEntryConfig(fixtureConfigPath, config, builtins, false, trusted);
        expect(Object.isFrozen(res)).toBe(true);
        expect(res.configPath).toBe(fixtureConfigPath);
        if (res.embedding.backend !== "onnx") throw new Error("expected onnx backend");
        expect(res.embedding.device).toBe("cpu");
        expect(res.embedding.fallback).toBe("error");
        expect(res.embedding.dtype).toBe("fp32");
        expect(res.embedding.allowDownload).toBe(false);
        expect(res.embedding.intraOpNumThreads).toBe(1);
        expect(res.embedding.batchSize).toBe(1);
        expect(res.embedding.maxBatchTokens).toBe(4096);
        expect(Object.isFrozen(res.config)).toBe(true);
        expect(Object.isFrozen(res.embedding)).toBe(true);
        expect(Object.isFrozen(res.embedding.source)).toBe(true);
        expect(Object.isFrozen(res.embedding.recipe)).toBe(true);
        expect(res.config.models?.embed).toBe(res.embedding);
        expect(typeof res.digest).toBe("string");
        expect(res.digest.length).toBeGreaterThan(0);
        expect(calls.length).toBe(2);
        expect(calls[0]).toBe(fixtureConfigPath);
        expect(calls[1]).toBe(res.digest);
        const res2 = approveEntryConfig(fixtureConfigPath, config, builtins, false, trusted);
        expect(res2.digest).toBe(res.digest);
        expect(res2.embedding).not.toBe(res.embedding);
        expect(res2.embedding).toEqual(res.embedding);
      },
    },
    {
      name: "denied local noninteractive actionable both cases",
      run() {
        const config: CollectionConfig = { collections: {}, models: { embed: { backend: "onnx", source: { kind: "local", path: "/fixture/model.onnx" }, recipe: makeOnnxConfig().recipe } } };
        let err: unknown;
        try { approveEntryConfig(fixtureConfigPath, config, builtins, false, () => false); } catch (e) { err = e; }
        expect(err).toBeDefined();
        const msg = String(err instanceof Error ? err.message : err);
        expect(/trust|approv/i.test(msg)).toBe(true);

        let err2: unknown;
        try { approveEntryConfig(fixtureConfigPath, config, builtins, true, () => false); } catch (e) { err2 = e; }
        expect(err2).toBeDefined();
        const msg2 = String(err2 instanceof Error ? err2.message : err2);
        expect(/trust|approv/i.test(msg2)).toBe(true);
      },
    },
    {
      name: "relative LOCAL source resolves dirname configPath",
      run() {
        const config: CollectionConfig = { collections: {}, models: { embed: { backend: "onnx", source: mutableLocalSource("./weights"), recipe: makeOnnxConfig().recipe } } };
        const res = approveEntryConfig(fixtureConfigPath, config, builtins, false, () => true);
        if (res.embedding.backend !== "onnx") throw new Error("expected onnx");
        if (res.embedding.source.kind !== "local") throw new Error("expected local");
        expect(res.embedding.source.path).toBe(resolve(dirname(fixtureConfigPath), "./weights"));
        expect(res.config.models?.embed).toBe(res.embedding);
      },
    },
    {
      name: "global path allowed without trustedCheck and checker throws",
      run() {
        const config: CollectionConfig = { collections: { docs: { path: "/global/docs", pattern: "**/*.md" } }, models: { embed: { backend: "onnx", source: { kind: "huggingface", model: "m", revision: "b".repeat(40) }, recipe: makeOnnxConfig().recipe } } };
        const checker = () => { throw new Error("checker_should_not_run"); };
        const res = approveEntryConfig(globalConfigPath, config, builtins, false, checker);
        expect(res.configPath).toBe(globalConfigPath);
        expect(isLocalConfigPath(globalConfigPath)).toBe(false);
      },
    },
    {
      name: "caller original JSON unchanged and not frozen",
      run() {
        const config: CollectionConfig = { collections: { a: { path: "./p", pattern: "**/*.md" } }, models: { embed: { backend: "onnx", source: { kind: "huggingface", model: "m", revision: "c".repeat(40) }, recipe: makeOnnxConfig().recipe } } };
        const original = JSON.stringify(config);
        const res = approveEntryConfig(fixtureConfigPath, config, builtins, false, () => true);
        expect(JSON.stringify(config)).toBe(original);
        expect(Object.isFrozen(config)).toBe(false);
        expect(res.config).not.toBe(config);
      },
    },
    {
      name: "after approval mutable aliases changed capture unchanged source/recipe/context",
      run() {
        const baseRecipe = makeOnnxConfig().recipe;
        const mutableSource = mutableLocalSource("/fixture/model.onnx");
        const mutableRecipe = { ...baseRecipe, queryPrefix: "orig:" };
        const context = { x: "orig" };
        const config: CollectionConfig = { collections: { docs: { path: "./docs", pattern: "**/*.md", context } }, models: { embed: { backend: "onnx", source: mutableSource, recipe: mutableRecipe } } };
        const res = approveEntryConfig(fixtureConfigPath, config, builtins, false, () => true);
        if (res.embedding.backend !== "onnx") throw new Error("expected onnx");
        mutableSource.path = "/other/weights";
        mutableRecipe.queryPrefix = "changed:";
        context.x = "changed";
        expect(res.config.collections?.docs?.path).toBe("./docs");
        expect(res.embedding.source).not.toBe(mutableSource);
        if (res.embedding.source.kind !== "local") throw new Error("expected local source");
        expect(res.embedding.source.path).toBe("/fixture/model.onnx");
        expect(res.embedding.recipe.queryPrefix).toBe("orig:");
        expect(res.config.collections?.docs?.context).toEqual({ x: "orig" });
        expect(Object.isFrozen(res.config.collections?.docs?.context)).toBe(true);
        expect(Object.isFrozen(context)).toBe(false);
        expect(Object.isFrozen(mutableSource)).toBe(false);
        expect(Object.isFrozen(mutableRecipe)).toBe(false);
      },
    },
    {
      name: "callback mutates mutable source model and recipe prefix before approval captured",
      run() {
        const mutableSource: { kind: "huggingface"; model: string; revision: string } = { kind: "huggingface", model: "fixture/model", revision: "e".repeat(40) };
        const baseRecipe = makeOnnxConfig().recipe;
        const mutableRecipe = { ...baseRecipe, queryPrefix: "base:" };
        const config: CollectionConfig = { collections: {}, models: { embed: { backend: "onnx", source: mutableSource, recipe: mutableRecipe } } };
        let capturedDigest = "";
        const trusted = (_p: string, d: string) => { capturedDigest = d; mutableSource.model = "mutated"; mutableRecipe.queryPrefix = "mut:"; return true; };
        const res = approveEntryConfig(fixtureConfigPath, config, builtins, false, trusted);
        if (res.embedding.backend !== "onnx") throw new Error("expected onnx");
        if (res.embedding.source.kind !== "huggingface") throw new Error("expected huggingface source");
        expect(res.embedding.source.model).toBe("fixture/model");
        expect(res.embedding.recipe.queryPrefix).toBe("base:");
        expect(capturedDigest).toBe(res.digest);
        expect(res.digest).not.toBe("");
      },
    },
    {
      name: "selected ONNX device CPU and no GGUF substitute",
      run() {
        const config: CollectionConfig = { collections: {}, models: { embed: { backend: "onnx", source: { kind: "huggingface", model: "m", revision: "f".repeat(40) }, recipe: makeOnnxConfig().recipe, device: "cpu" } } };
        const res = approveEntryConfig(fixtureConfigPath, config, builtins, false, () => true);
        if (res.embedding.backend !== "onnx") throw new Error("expected onnx");
        expect(res.embedding.device).toBe("cpu");
        expect(res.config.models?.embed).toBe(res.embedding);
      },
    },
    {
      name: "legacy builtin missing model returns gguf default string",
      run() {
        const config: CollectionConfig = { collections: {}, models: {} };
        const res = approveEntryConfig(fixtureConfigPath, config, builtins, false, () => { throw new Error("builtins must not need approval"); });
        if (res.embedding.backend !== "gguf") throw new Error("expected gguf default");
        expect(res.embedding.model).toBe(builtins.embed);
        expect(res.config.models?.embed).toBe(builtins.embed);
      },
    },
    {
      name: "approved custom GGUF selects string not object",
      run() {
        const config: CollectionConfig = { collections: {}, models: { embed: "custom.gguf" } };
        const res = approveEntryConfig(fixtureConfigPath, config, builtins, false, () => true);
        expect(res.embedding.backend).toBe("gguf");
        if (res.embedding.backend === "gguf") {
          expect(res.embedding.model).toBe("custom.gguf");
        }
        expect(res.config.models?.embed).toBe("custom.gguf");
      },
    },
  ];

  for (const c of cases) {
    test(c.name, () => c.run());
  }

  test("T14c1 CASE1 preserved exact source string no inferred save", () => {
    const declared = makeOnnxConfig();
    if (declared.source.kind !== "huggingface") throw new Error("fixture");
    const expectedSource = declared.source.model;
    const config: CollectionConfig = { collections: {}, models: { embed: declared } };
    setConfigSource({ config });
    const before = JSON.stringify(config.models);
    try {
      expect(typeof resolveEmbedModelForCli()).toBe("string");
      expect(resolveEmbedModelForCli()).toBe(expectedSource);
      expect(typeof resolveGenerateModelForCli()).toBe("string");
      expect(resolveGenerateModelForCli().length).toBeGreaterThan(0);
      expect(typeof resolveRerankModelForCli()).toBe("string");
      expect(resolveRerankModelForCli().length).toBeGreaterThan(0);
      expect(JSON.stringify(loadConfig().models)).toBe(before);
      expect(JSON.stringify(config.models)).toBe(before);
    } finally { setConfigSource(undefined); }
  });

  test("T14c1 CASE2 empty config no inferred save undefined", () => {
    const config: CollectionConfig = { collections: {} };
    setConfigSource({ config });
    const before = JSON.stringify(config);
    try {
      expect(typeof resolveEmbedModelForCli()).toBe("string");
      expect(resolveEmbedModelForCli().length).toBeGreaterThan(0);
      expect(typeof resolveGenerateModelForCli()).toBe("string");
      expect(resolveGenerateModelForCli().length).toBeGreaterThan(0);
      expect(typeof resolveRerankModelForCli()).toBe("string");
      expect(resolveRerankModelForCli().length).toBeGreaterThan(0);
      expect(loadConfig().models).toBeUndefined();
      expect(JSON.stringify(config)).toBe(before);
    } finally { setConfigSource(undefined); }
  });
});
