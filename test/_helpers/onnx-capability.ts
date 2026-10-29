import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { createArtifactIO } from "../../src/embeddings/artifact-io.js";
import { resolveEmbeddingConfig } from "../../src/embeddings/config.js";
import { createOnnxEmbeddingCapability } from "../../src/embeddings/onnx.js";
import { makeOnnxConfig } from "./embedding-config.js";
import { deferred } from "./deferred.js";
import type { ArtifactSnapshot } from "../../src/embeddings/artifacts.js";
import type { EmbeddingCapability, OnnxEmbeddingConfig } from "../../src/embeddings/types.js";
import type { EmbeddingModelOutput, EmbeddingTokenizer } from "../../src/embeddings/recipe.js";
import type { OnnxLoadOptions, OnnxRuntimePort, OnnxRuntimeSession } from "../../src/embeddings/onnx-runtime-types.js";

export const recipe = {
  version: "qmd-embedding-v1", pooling: "mean", normalization: "none",
  queryPrefix: "q: ", documentPrefix: "d: ", maxInputTokens: 128, titleBudgetTokens: 4, overlapTokens: 2,
} as const;

function graph(): Buffer { return Buffer.from([8, 10, 58, 0]); }
function modelFiles(): readonly (readonly [string, Buffer])[] {
  return [
    ["config.json", Buffer.from(JSON.stringify({ model_type: "bert", hidden_size: 2, max_position_embeddings: 512 }))],
    ["tokenizer_config.json", Buffer.from('{"tokenizer_class":"BertTokenizer"}')],
    ["tokenizer.json", Buffer.from('{"model":{"type":"WordPiece","vocab":{"[UNK]":0},"unk_token":"[UNK]"}}')],
    ["onnx/model.onnx", graph()],
  ];
}
function tokenLength(text: string): number { return Math.max(1, text.length) + 2; }

interface PortObservations {
  readonly loads: Array<Readonly<{ snapshot: ArtifactSnapshot; options: OnnxLoadOptions }>>;
  readonly runs: string[][];
  readonly tokenizations: Array<Readonly<{ text: string; special: boolean }>>;
  readonly diagnostics: Array<Readonly<{ code: string; requestedDevice: string; reason: string }>>;
  readonly counts: { releases: number; closes: number };
}
interface FakePort extends OnnxRuntimePort { readonly observations: PortObservations }
interface PortOptions {
  readonly platform?: NodeJS.Platform;
  readonly arch?: string;
  readonly loadFailure?: Error;
  readonly classifyLoadFailure?: boolean;
  readonly userRunGate?: ReturnType<typeof deferred<void>>;
  readonly userRunStarted?: ReturnType<typeof deferred<void>>;
  readonly userReleaseGate?: ReturnType<typeof deferred<void>>;
  readonly userReleaseStarted?: ReturnType<typeof deferred<void>>;
  readonly userRunFailure?: Error;
}
function fakePort(dimensions = 2, options: PortOptions = {}): FakePort {
  const observations: PortObservations = { loads: [], runs: [], tokenizations: [], diagnostics: [], counts: { releases: 0, closes: 0 } };
  const tokenizer: EmbeddingTokenizer = {
    async encode(text, addSpecialTokens) {
      observations.tokenizations.push({ text, special: addSpecialTokens });
      return Array.from({ length: addSpecialTokens ? tokenLength(text) : Math.max(1, text.length) }, (_, index) => index);
    },
    async decode(tokens) { return tokens.map(token => String.fromCharCode(65 + token % 26)).join(""); },
  };
  const load = async (snapshot: ArtifactSnapshot, loadOptions: OnnxLoadOptions): Promise<OnnxRuntimeSession> => {
    observations.loads.push({ snapshot, options: loadOptions });
    if (observations.loads.length === 1 && options.loadFailure !== undefined) throw options.loadFailure;
    return {
      tokenizer,
      async run(texts) {
        const runIndex = observations.runs.length;
        observations.runs.push([...texts]);
        if (runIndex > 0) {
          options.userRunStarted?.resolve();
          await options.userRunGate?.promise;
          if (options.userRunFailure !== undefined) throw options.userRunFailure;
        }
        const lengths = texts.map(tokenLength), sequence = Math.max(...lengths);
        const mask = lengths.map(length => Array.from({ length: sequence }, (_, index) => index < length ? 1 : 0));
        const data = new Float32Array(texts.length * sequence * dimensions);
        for (let batch = 0; batch < texts.length; batch++) {
          const length = lengths[batch];
          const text = texts[batch];
          if (length === undefined || text === undefined) throw new Error("Invalid fake batch");
          for (let token = 0; token < sequence; token++) {
            for (let dimension = 0; dimension < dimensions; dimension++) {
              const index = batch * sequence * dimensions + token * dimensions + dimension;
              data[index] = token < length ? text.length + batch + token / 10 + dimension / 100 : 1000;
            }
          }
        }
        const output: EmbeddingModelOutput = { last_hidden_state: { data, dims: [texts.length, sequence, dimensions] } };
        return {
          output, mask,
          async release() {
            observations.counts.releases++;
            if (runIndex > 0) {
              options.userReleaseStarted?.resolve();
              await options.userReleaseGate?.promise;
            }
          },
        };
      },
      async close() { observations.counts.closes++; },
    };
  };
  return {
    platform: options.platform ?? "linux", arch: options.arch ?? "x64", load,
    isProviderInitializationError: error => options.classifyLoadFailure === true && error === options.loadFailure,
    diagnostic: event => { observations.diagnostics.push(event); },
    observations,
  };
}

export interface Fixture {
  readonly local: string;
  readonly work: string;
  readonly port: (dimensions?: number, options?: PortOptions) => FakePort;
  readonly capability: (runtime: FakePort, overrides?: Partial<OnnxEmbeddingConfig>) => EmbeddingCapability;
}

export async function withFixture(run: (fixture: Fixture) => Promise<void>): Promise<void> {
  const root = await mkdtemp(resolve(".tmp", "onnx-capability-"));
  const local = join(root, "source"), work = join(root, "work"), capabilities: EmbeddingCapability[] = [];
  try {
    for (const [path, bytes] of modelFiles()) {
      const destination = join(local, path);
      await mkdir(dirname(destination), { recursive: true });
      await writeFile(destination, bytes);
    }
    const port = (dimensions?: number, options?: PortOptions) => fakePort(dimensions, options);
    const capability = (runtime: FakePort, overrides: Partial<OnnxEmbeddingConfig> = {}) => {
      const value = resolveEmbeddingConfig(makeOnnxConfig({ ...overrides, recipe: overrides.recipe ?? recipe,
        source: overrides.source ?? { kind: "local", path: local } }), { defaultGgufModel: "legacy", hostParallelism: 8 });
      if (value.backend !== "onnx") throw new Error("Fixture config must be ONNX");
      const cap = createOnnxEmbeddingCapability(value, createArtifactIO(work), runtime);
      capabilities.push(cap);
      return cap;
    };
    await run({ local, work, port, capability });
  } finally {
    try { await Promise.all(capabilities.map(cap => cap.close())); }
    finally { await rm(root, { recursive: true, force: true }); }
  }
}
