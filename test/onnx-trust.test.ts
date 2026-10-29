import { describe, expect, test } from "vitest";
import { createHash } from "node:crypto";
import type { SensitiveSnapshot, BuiltinModels } from "../src/trust.js";
import { sensitiveDigest, decideLocalConfigGate, gatedModels } from "../src/trust.js";
import { resolveEmbeddingConfig } from "../src/embeddings/config.js";
import { makeOnnxConfig } from "./_helpers/embedding-config.js";
import type { ResolvedOnnxConfig, OnnxEmbeddingConfig } from "../src/embeddings/types.js";

const configPath = "/fixture/project/.qmd/index.yml";
const globalPath = "/fixture/home/.config/qmd/index.yml";

const builtins: BuiltinModels = {
  embed: "builtin.gguf",
  rerank: "builtin.gguf",
  generate: "builtin.gguf",
};

function snapshot(overrides: Partial<SensitiveSnapshot> = {}): SensitiveSnapshot {
  return {
    hooks: overrides.hooks ?? [],
    paths: overrides.paths ?? [],
    models: overrides.models ?? {},
  };
}

const ctx = { configPath, defaultGgufModel: "builtin.gguf", hostParallelism: 8 };

function resolved(overrides: Partial<OnnxEmbeddingConfig> = {}): ResolvedOnnxConfig {
  const cfg = resolveEmbeddingConfig(makeOnnxConfig(overrides), ctx);
  if (cfg.backend !== "onnx") throw new Error("fixture: expected onnx backend, got " + String(cfg.backend));
  return cfg;
}

// Legacy oracle: fixed SHA256 over old flat JSON representation.
const OLD_EMPTY = createHash("sha256").update(JSON.stringify({ hooks: [], paths: [], models: [] })).digest("hex");
const OLD_CUSTOM = createHash("sha256").update(JSON.stringify({ hooks: [], paths: [], models: [["embed", "custom.gguf"]] })).digest("hex");

describe("onnx-trust digest contract", () => {
  test("old empty format digest is exact legacy oracle", () => {
    expect(sensitiveDigest(snapshot(), configPath, builtins)).toBe(OLD_EMPTY);
  });

  test("old custom model format digest is exact legacy oracle", () => {
    expect(sensitiveDigest(snapshot({ models: { embed: "custom.gguf" } }), configPath, builtins)).toBe(OLD_CUSTOM);
  });

  test("missing and builtin model strings are equivalent empty digest", () => {
    const empty = sensitiveDigest(snapshot({ models: {} }), configPath, builtins);
    const builtin = sensitiveDigest(snapshot({ models: { embed: builtins.embed, rerank: builtins.rerank, generate: builtins.generate } }), configPath, builtins);
    expect(empty).toBe(builtin);
  });

  test("structured resolved config changes digest with source revision", () => {
    const base = resolved();
    const rev = resolved({ source: { kind: "huggingface", model: "fixture/model", revision: "a".repeat(40) } });
    const alt = resolved({ source: { kind: "huggingface", model: "fixture/model", revision: "b".repeat(40) } });
    expect(sensitiveDigest(snapshot({ models: { embed: base } }), configPath, builtins)).toBe(sensitiveDigest(snapshot({ models: { embed: rev } }), configPath, builtins));
    expect(sensitiveDigest(snapshot({ models: { embed: rev } }), configPath, builtins)).not.toBe(sensitiveDigest(snapshot({ models: { embed: alt } }), configPath, builtins));
  });

  test("structured digest changes with local abspath", () => {
    const rel = resolved({ source: { kind: "local", path: "fixture/model" } });
    const abs = resolved({ source: { kind: "local", path: "/fixture/project/model.onnx" } });
    expect(sensitiveDigest(snapshot({ models: { embed: rel } }), configPath, builtins)).not.toBe(sensitiveDigest(snapshot({ models: { embed: abs } }), configPath, builtins));
  });

  test("structured digest changes with recipe pooling cls and valid CLS index", () => {
    const mean = resolved({ recipe: { ...makeOnnxConfig().recipe, pooling: "mean", clsTokenIndex: undefined } });
    const cls = resolved({ recipe: { ...makeOnnxConfig().recipe, pooling: "cls", clsTokenIndex: 3 } });
    expect(sensitiveDigest(snapshot({ models: { embed: mean } }), configPath, builtins)).not.toBe(sensitiveDigest(snapshot({ models: { embed: cls } }), configPath, builtins));
  });

  test("structured digest changes with each recipe field one-at-time", () => {
    const base = resolved();
    const baseRecipe = makeOnnxConfig().recipe;
    const cases: readonly { name: string; overrides: Partial<OnnxEmbeddingConfig> }[] = [
      { name: "queryPrefix", overrides: { recipe: { ...baseRecipe, queryPrefix: "q:" } } },
      { name: "documentPrefix", overrides: { recipe: { ...baseRecipe, documentPrefix: "d:" } } },
      { name: "normalization none", overrides: { recipe: { ...baseRecipe, normalization: "none" } } },
      { name: "titleBudgetTokens", overrides: { recipe: { ...baseRecipe, titleBudgetTokens: 64 } } },
      { name: "overlapTokens", overrides: { recipe: { ...baseRecipe, overlapTokens: 32 } } },
      { name: "maxInputTokens", overrides: { recipe: { ...baseRecipe, maxInputTokens: 512 } } },
      { name: "pooling cls valid CLS", overrides: { recipe: { ...baseRecipe, pooling: "cls", clsTokenIndex: 2 } } },
    ];
    for (const c of cases) {
      const cfg = resolved(c.overrides);
      const s = snapshot({ models: { embed: cfg } });
      expect(sensitiveDigest(s, configPath, builtins)).not.toBe(sensitiveDigest(snapshot({ models: { embed: base } }), configPath, builtins));
    }
  });

  test("normalized defaults give same digest as omitted operational defaults", () => {
    const omitted = resolved();
    const explicit = resolved({ device: "cpu", fallback: "error", dtype: "fp32", allowDownload: false, intraOpNumThreads: 1, batchSize: 1, maxBatchTokens: 4096 });
    expect(sensitiveDigest(snapshot({ models: { embed: omitted } }), configPath, builtins)).toBe(sensitiveDigest(snapshot({ models: { embed: explicit } }), configPath, builtins));
  });

  test("reordered recipe property keys give same digest (already resolved, spread reorder)", () => {
    const a = resolved();
    const { version, pooling, normalization, queryPrefix, documentPrefix, maxInputTokens, titleBudgetTokens, overlapTokens, clsTokenIndex, sentenceOutput } = a.recipe;
    const recipeReordered = { sentenceOutput, clsTokenIndex, overlapTokens, titleBudgetTokens, maxInputTokens, documentPrefix, queryPrefix, normalization, pooling, version };
    const b = { ...a, recipe: recipeReordered };
    expect(JSON.stringify(a.recipe)).not.toBe(JSON.stringify(recipeReordered));
    expect(sensitiveDigest(snapshot({ models: { embed: a } }), configPath, builtins)).toBe(sensitiveDigest(snapshot({ models: { embed: b } }), configPath, builtins));
  });

  test("structured digest changes with source HF repo, revision, local absolute, device, fallback, dtype, dimensions, allowDownload, threads, batch, maxBatchTokens", () => {
    const base = resolved();
    const cases: readonly { name: string; overrides: Partial<OnnxEmbeddingConfig> }[] = [
      { name: "source model id", overrides: { source: { kind: "huggingface", model: "fixture/encoder", revision: "a".repeat(40) } } },
      { name: "source rev", overrides: { source: { kind: "huggingface", model: "fixture/model", revision: "b".repeat(40) } } },
      { name: "source local abs", overrides: { source: { kind: "local", path: "/fixture/project/model.onnx" } } },
      { name: "device cuda", overrides: { device: "cuda" } },
      { name: "fallback cpu", overrides: { fallback: "cpu" } },
      { name: "dtype q8", overrides: { dtype: "q8" } },
      { name: "dimensions 768", overrides: { dimensions: 768 } },
      { name: "allowDownload true", overrides: { allowDownload: true } },
      { name: "threads 8", overrides: { intraOpNumThreads: 8 } },
      { name: "batch 4", overrides: { batchSize: 4 } },
      { name: "maxBatchTokens 8192", overrides: { maxBatchTokens: 8192 } },
    ];
    for (const c of cases) {
      const cfg = resolved(c.overrides);
      const s = snapshot({ models: { embed: cfg } });
      expect(sensitiveDigest(s, configPath, builtins)).not.toBe(sensitiveDigest(snapshot({ models: { embed: base } }), configPath, builtins));
    }
  });

  test("all ONNX always custom gated even named builtin repo; entry uri is string display label", () => {
    const builtinRepo = resolved({ source: { kind: "huggingface", model: "builtin.gguf", revision: "a".repeat(40) } });
    const gated = gatedModels({ embed: builtinRepo }, builtins);
    expect(gated.length).toBe(1);
    const item = gated[0];
    if (!item) throw new Error("fixture: expected gated entry");
    expect(item.slot).toBe("embed");
    expect(typeof item.uri).toBe("string");
  });

  test("a legacy string equal to an ONNX display label cannot reuse structured approval", () => {
    const embed = resolved(), item = gatedModels({ embed }, builtins)[0];
    if (!item) throw new Error("fixture gated model missing");
    expect(sensitiveDigest(snapshot({ models: { embed } }), configPath, builtins))
      .not.toBe(sensitiveDigest(snapshot({ models: { embed: item.uri } }), configPath, builtins));
  });

  test("mixed structured and legacy model roles retain tagged serialization and old URI bytes", () => {
    const embed = resolved(), item = gatedModels({ embed }, builtins)[0];
    if (!item) throw new Error("fixture gated model missing");
    const expected = createHash("sha256").update(JSON.stringify({ hooks: [], paths: [], models: [
      ["embed", { backend: "onnx", configuration: item.uri }], ["rerank", "custom-rerank.gguf"],
    ] })).digest("hex");
    const mixed = sensitiveDigest(snapshot({ models: { embed, rerank: "custom-rerank.gguf" } }), configPath, builtins);
    expect(mixed).toBe(expected);
    expect(mixed).not.toBe(sensitiveDigest(snapshot({ models: { embed } }), configPath, builtins));
    expect(mixed).not.toBe(sensitiveDigest(snapshot({ models: { rerank: "custom-rerank.gguf" } }), configPath, builtins));
    for (const uri of [item.uri.slice(0, -1), `"${item.uri}`]) {
      expect(sensitiveDigest(snapshot({ models: { embed: uri } }), configPath, builtins))
        .not.toBe(sensitiveDigest(snapshot({ models: { embed } }), configPath, builtins));
    }
  });

  test("gatedModels never asserts exact entry object shape; length 1 only", () => {
    const gated = gatedModels({ embed: "custom.gguf" }, builtins);
    expect(gated.length).toBe(1);
    const item = gated[0];
    if (!item) throw new Error("fixture: expected gated entry");
    expect(typeof item.slot).toBe("string");
    expect(typeof item.uri).toBe("string");
  });
});

describe("decideLocalConfigGate contract", () => {
  const noEnv: NodeJS.ProcessEnv = {};
  const untrusted = () => false;
  const trusted = () => true;

  test("local config untrusted checker false with env:{} skips", () => {
    const decision = decideLocalConfigGate({ configPath, snapshot: snapshot({ paths: [{ collection: "out", path: "/etc" }] }), builtins, isInteractive: false, env: noEnv, trustedCheck: untrusted });
    expect(decision.action).toBe("skip");
  });

  test("trusted checker true runs", () => {
    const decision = decideLocalConfigGate({ configPath, snapshot: snapshot({ paths: [{ collection: "out", path: "/etc" }] }), builtins, isInteractive: false, env: noEnv, trustedCheck: trusted });
    expect(decision.action).toBe("run");
  });

  test("global path always runs regardless of untrusted checker", () => {
    const decision = decideLocalConfigGate({ configPath: globalPath, snapshot: snapshot({ paths: [{ collection: "out", path: "/etc" }] }), builtins, isInteractive: false, env: noEnv, trustedCheck: untrusted });
    expect(decision.action).toBe("run");
  });

  test("rearm via digest change not authorization sideeffects", () => {
    const digestOld = sensitiveDigest(snapshot({ paths: [{ collection: "docs", path: "./docs" }] }), configPath, builtins);
    const decision = decideLocalConfigGate({ configPath, snapshot: snapshot({ paths: [{ collection: "docs", path: "/etc" }] }), builtins, isInteractive: false, env: noEnv, trustedCheck: (_p, d) => d === digestOld });
    expect(decision.action).toBe("skip");
  });

  test("structured model snapshot runs when approved by digest", () => {
    const base = resolved();
    const s = snapshot({ models: { embed: base  } });
    const digest = sensitiveDigest(s, configPath, builtins);
    const decision = decideLocalConfigGate({ configPath, snapshot: s, builtins, isInteractive: false, env: noEnv, trustedCheck: (_p, d) => d === digest });
    expect(decision.action).toBe("run");
  });
});
