import { createHash } from "node:crypto";
import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { isAbsolute, join, resolve, sep } from "node:path";
import { afterEach, expect, test } from "vitest";
import { createArtifactManifest } from "../src/embeddings/identity.js";
import { parseEmbeddingExportProfile } from "../src/embeddings/artifact-profile.js";
import { createTransformersRuntime } from "../src/embeddings/onnx-transformers.js";
import type { ArtifactSnapshot } from "../src/embeddings/artifacts.js";
import type { OnnxLoadOptions } from "../src/embeddings/onnx-runtime-types.js";
import type { ArtifactEntry } from "../src/embeddings/types.js";

interface Fixture {
  readonly directory: string;
  readonly snapshot: ArtifactSnapshot;
}
interface RuntimeFake {
  readonly module: unknown;
  readonly tokenizerArguments: { json: unknown; config: unknown }[];
  readonly modelCalls: { directory: string; options: unknown }[];
  readonly autoTokenizerReads: () => number;
  readonly disposeCalls: () => number;
}
const scratchDirectories: string[] = [];
function createFixture(graphPath = "onnx/model.onnx"): Fixture {
  const directory = mkdtempSync(resolve(".tmp", "onnx-transformers-"));
  scratchDirectories.push(directory);
  const configText = JSON.stringify({ model_type: "bert", hidden_size: 192, max_position_embeddings: 32 });
  const tokenizerConfigText = JSON.stringify({ tokenizer_class: "BertTokenizer", model_max_length: 32, unk_token: "[UNK]" });
  const tokenizerDataText = JSON.stringify({ model: { type: "WordPiece", vocab: { "[UNK]": 0, hello: 1 } }, unk_token: "[UNK]" });
  const metadata: readonly (readonly [string, string])[] = [
    ["config.json", configText], ["tokenizer_config.json", tokenizerConfigText], ["tokenizer.json", tokenizerDataText],
  ];
  const entries: ArtifactEntry[] = metadata.map(([path, contents]) => {
    const bytes = Buffer.from(contents, "utf8");
    writeFileSync(join(directory, path), bytes);
    return { path, bytes: bytes.length, sha256: createHash("sha256").update(bytes).digest("hex") };
  });
  const graph = Buffer.from([0x4f, 0x4e, 0x4e, 0x58]);
  const graphFile = join(directory, ...graphPath.split("/"));
  mkdirSync(join(directory, ...graphPath.split("/").slice(0, -1)), { recursive: true });
  writeFileSync(graphFile, graph);
  entries.push({ path: graphPath, bytes: graph.length, sha256: createHash("sha256").update(graph).digest("hex") });
  const config = JSON.parse(configText), tokenizerConfig = JSON.parse(tokenizerConfigText), tokenizerData = JSON.parse(tokenizerDataText);
  const profile = parseEmbeddingExportProfile(config, tokenizerConfig, tokenizerData);
  return {
    directory,
    snapshot: {
      directory,
      manifest: createArtifactManifest({ kind: "local", path: directory }, entries),
      profile,
      release: async () => {},
    },
  };
}
function fakeRuntime(overrides: { unsafeEnvironment?: boolean; failModelLoad?: boolean; invalidSession?: boolean } = {}): RuntimeFake {
  const tokenizerArguments: { json: unknown; config: unknown }[] = [];
  const modelCalls: { directory: string; options: unknown }[] = [];
  let autoTokenizerReads = 0, disposeCount = 0;
  const env = {
    version: "4.3.1",
    allowRemoteModels: true, allowLocalModels: !overrides.unsafeEnvironment, useFS: true,
    useFSCache: true, useBrowserCache: false, useCustomCache: false,
    experimental_useCrossOriginStorage: false,
  };
  function BertTokenizer(json: unknown, config: unknown) {
    tokenizerArguments.push({ json, config });
    const wrap = (values: readonly bigint[]) => {
      const tensor = { type: "int64", data: new BigInt64Array(values), dims: [1, values.length], location: "cpu", dispose() {} };
      return { ort_tensor: tensor, dispose: () => tensor.dispose() };
    };
    return Object.assign(
      (_texts: unknown, _options: unknown) => ({ input_ids: wrap([1n, 2n]), attention_mask: wrap([1n, 1n]) }),
      { encode: (_text: string, _options: unknown) => [1, 2], decode: (_tokens: readonly number[], _options: unknown) => "decoded" },
    );
  }
  const BertModel = {
    async from_pretrained(directory: string, options: unknown): Promise<unknown> {
      modelCalls.push({ directory, options });
      if (overrides.failModelLoad) throw Object.assign(new Error("private model path"), { path: "private/path", cause: new Error("private cause") });
      const session = { inputNames: ["input_ids", "attention_mask"], async run() { return {}; } };
      if (overrides.invalidSession) {
        return {
          get sessions() { throw new Error("private session detail"); },
          dispose() { disposeCount++; },
        };
      }
      return { sessions: { model: session }, dispose() { disposeCount++; } };
    },
  };
  function Tensor(_type: string, data: BigInt64Array, dimensions: readonly number[]) {
    return { type: "int64", data, dims: [...dimensions], location: "cpu", dispose() {} };
  }
  const module = Object.defineProperties({ env, BertTokenizer, BertModel, Tensor }, {
    AutoTokenizer: { get() { autoTokenizerReads++; throw new Error("AutoTokenizer must not be used"); } },
  });
  return { module, tokenizerArguments, modelCalls, autoTokenizerReads: () => autoTokenizerReads, disposeCalls: () => disposeCount };
}
function loadOptions(dtype: OnnxLoadOptions["dtype"] = "fp32"): OnnxLoadOptions {
  return { device: "cpu", providers: ["cpu"], dtype, intraOpNumThreads: 2,
    sequential: true, memoryPattern: true, localFilesOnly: true };
}
function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}
function record(value: unknown): Record<string, unknown> {
  if (!isRecord(value)) throw new Error("Expected object");
  return value;
}
afterEach(() => {
  for (const directory of scratchDirectories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

test("runtime construction is lazy; load uses direct BERT files and owned CPU options", async () => {
  const fixture = createFixture(), fake = fakeRuntime(); let imports = 0;
  expect(isAbsolute(fixture.snapshot.directory)).toBe(true);
  const runtime = createTransformersRuntime(async () => { imports++; return fake.module; });
  expect(imports).toBe(0);
  const session = await runtime.load(fixture.snapshot, loadOptions());
  expect(imports).toBe(1);
  expect(fake.tokenizerArguments).toHaveLength(1);
  expect(record(fake.tokenizerArguments[0]?.config).tokenizer_class).toBe("BertTokenizer");
  expect(fake.modelCalls).toHaveLength(1);
  const call = fake.modelCalls[0];
  expect(call?.directory).toBe(fixture.directory);
  const options = record(call?.options);
  expect(options.local_files_only).toBe(true);
  expect(options.device).toBe("cpu"); expect(options.dtype).toBe("fp32");
  expect(options.subfolder).toBe("onnx"); expect(options.model_file_name).toBe("model");
  expect(options.use_external_data_format).toBe(false);
  expect(options.config).toEqual({ model_type: "bert", hidden_size: 192, max_position_embeddings: 32 });
  expect(isAbsolute(String(options.cache_dir))).toBe(true);
  expect(String(options.cache_dir).startsWith(fixture.directory + sep)).toBe(true);
  const sessionOptions = record(options.session_options);
  expect(sessionOptions.executionProviders).toEqual(["cpu"]);
  expect(sessionOptions.intraOpNumThreads).toBe(2);
  expect(sessionOptions.executionMode).toBe("sequential");
  expect(sessionOptions.enableMemPattern).toBe(true);
  expect(sessionOptions.freeDimensionOverrides).toEqual({});
  expect(Object.hasOwn(sessionOptions, "externalData")).toBe(false);
  expect(fake.autoTokenizerReads()).toBe(0);
  await session.close(); expect(fake.disposeCalls()).toBe(1);
});

test("q8 selects only its quantized graph and explicit q8 model option", async () => {
  const fixture = createFixture("onnx/model_quantized.onnx"), fake = fakeRuntime();
  const runtime = createTransformersRuntime(async () => fake.module);
  const session = await runtime.load(fixture.snapshot, loadOptions("q8"));
  expect(record(fake.modelCalls[0]?.options).dtype).toBe("q8");
  await session.close();
});

test("missing selected graph and changed metadata bytes fail before model loading", async () => {
  const fixture = createFixture(), fake = fakeRuntime(), runtime = createTransformersRuntime(async () => fake.module);
  await expect(runtime.load(fixture.snapshot, loadOptions("q8"))).rejects.toThrow(/graph|artifact|manifest/i);
  expect(fake.modelCalls).toHaveLength(0);
  writeFileSync(join(fixture.directory, "tokenizer.json"), "{}", "utf8");
  await expect(runtime.load(fixture.snapshot, loadOptions())).rejects.toThrow(/metadata|integrity|artifact|manifest/i);
  expect(fake.modelCalls).toHaveLength(0);
});

test("unsafe process environment fails before model loading", async () => {
  const fixture = createFixture(), fake = fakeRuntime({ unsafeEnvironment: true });
  const runtime = createTransformersRuntime(async () => fake.module);
  await expect(runtime.load(fixture.snapshot, loadOptions())).rejects.toThrow(/environment|filesystem|cache/i);
  expect(fake.modelCalls).toHaveLength(0);
});

test("loader errors are sanitized and a returned model is disposed if session admission fails", async () => {
  const fixture = createFixture(), badLoad = fakeRuntime({ failModelLoad: true });
  const runtime = createTransformersRuntime(async () => badLoad.module);
  const failure: unknown = await runtime.load(fixture.snapshot, loadOptions()).catch((error: unknown) => error);
  expect(failure).toBeInstanceOf(Error);
  if (!(failure instanceof Error)) throw new Error("Expected sanitized Error");
  expect(failure.message).toBe("ONNX model loading failed");
  expect(failure.message).not.toContain("private"); expect("cause" in failure).toBe(false); expect("path" in failure).toBe(false);
  expect(badLoad.modelCalls).toHaveLength(1);
  const badSession = fakeRuntime({ invalidSession: true }), retry = createTransformersRuntime(async () => badSession.module);
  await expect(retry.load(fixture.snapshot, loadOptions())).rejects.toThrow(/session|contract/i);
  expect(badSession.disposeCalls()).toBe(1);
});
