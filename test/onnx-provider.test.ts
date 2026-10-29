import { expect, test } from "vitest";
import { onnxLoadOptions } from "../src/embeddings/onnx-provider.js";
import { resolveEmbeddingConfig } from "../src/embeddings/config.js";
import { makeOnnxConfig } from "./_helpers/embedding-config.js";
import type { ResolvedOnnxConfig } from "../src/embeddings/types.js";
import type { OnnxLoadOptions } from "../src/embeddings/onnx-runtime-types.js";

function config(device: OnnxLoadOptions["device"]): ResolvedOnnxConfig {
  const parsed = resolveEmbeddingConfig(makeOnnxConfig({ device, dtype: "q8", intraOpNumThreads: 4, fallback: "cpu" }),
    { defaultGgufModel: "legacy", hostParallelism: 8 });
  if (parsed.backend !== "onnx") throw new Error("Wrong fixture backend");
  return parsed;
}
interface Target { readonly platform: NodeJS.Platform; readonly arch: string; readonly device: OnnxLoadOptions["device"] }
const targets: readonly Target[] = [
  { platform: "linux", arch: "x64", device: "cpu" }, { platform: "linux", arch: "arm64", device: "cpu" },
  { platform: "win32", arch: "x64", device: "cpu" }, { platform: "win32", arch: "arm64", device: "cpu" },
  { platform: "darwin", arch: "x64", device: "cpu" }, { platform: "darwin", arch: "arm64", device: "cpu" },
  { platform: "linux", arch: "x64", device: "cuda" },
  { platform: "win32", arch: "x64", device: "dml" }, { platform: "win32", arch: "arm64", device: "dml" },
  { platform: "darwin", arch: "x64", device: "coreml" }, { platform: "darwin", arch: "arm64", device: "coreml" },
];
test.each(targets)("owns explicit options for targeted $platform/$arch/$device", target => {
  const options = onnxLoadOptions(config(target.device), target);
  expect(options).toEqual({ device: target.device, providers: [target.device], dtype: "q8", intraOpNumThreads: 4,
    sequential: true, memoryPattern: target.device !== "dml", localFilesOnly: true });
  expect(Object.isFrozen(options)).toBe(true); expect(Object.isFrozen(options.providers)).toBe(true);
});
const unsupported: readonly Target[] = [
  { platform: "freebsd", arch: "x64", device: "cpu" }, { platform: "android", arch: "arm64", device: "cpu" },
  { platform: "linux", arch: "ia32", device: "cpu" }, { platform: "darwin", arch: "riscv64", device: "cpu" },
  { platform: "darwin", arch: "x64", device: "cuda" }, { platform: "win32", arch: "x64", device: "cuda" },
  { platform: "linux", arch: "arm64", device: "cuda" }, { platform: "linux", arch: "x64", device: "dml" },
  { platform: "darwin", arch: "arm64", device: "dml" }, { platform: "linux", arch: "arm64", device: "coreml" },
  { platform: "win32", arch: "x64", device: "coreml" },
];
test.each(unsupported)("rejects unsupported target $platform/$arch/$device without implicit fallback", target => {
  expect(() => onnxLoadOptions(config(target.device), target)).toThrow(/target|platform|unsupported/i);
});
test("a caller-controlled CPU retry leaves the requested config and model controls unchanged", () => {
  const selected = config("cuda"), host: Target = { platform: "linux", arch: "x64", device: "cuda" };
  const before = JSON.stringify(selected);
  const options = onnxLoadOptions(selected, host, "cpu");
  expect(options.providers).toEqual(["cpu"]); expect(options.dtype).toBe("q8");
  expect(options.intraOpNumThreads).toBe(4); expect(JSON.stringify(selected)).toBe(before); expect(selected.device).toBe("cuda");
});
