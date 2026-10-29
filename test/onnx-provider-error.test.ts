import { expect, test } from "vitest";
import { isOnnxProviderInitializationError } from "../src/embeddings/onnx-provider-error.js";

type Accelerator = "cuda" | "dml" | "coreml";
const providerCases: readonly { readonly device: Accelerator }[] = [{ device: "cuda" }, { device: "dml" }, { device: "coreml" }];
test.each(providerCases)("recognizes pinned ORT initialization errors for $device only", ({ device }) => {
  const error = new Error(`no available backend found. ERR: [${device}] backend not found.`);
  expect(isOnnxProviderInitializationError(error, device)).toBe(true);
});
test.each(providerCases)("recognizes pinned Transformers unsupported-provider errors for $device", ({ device }) => {
  const error = new Error(`Unsupported device: "${device}". Should be one of: cpu, wasm.`);
  expect(isOnnxProviderInitializationError(error, device)).toBe(true);
});
const mismatchCases: readonly { readonly requested: Accelerator; readonly actual: Accelerator }[] = [
  { requested: "cuda", actual: "dml" }, { requested: "dml", actual: "cuda" },
];
test.each(mismatchCases)("rejects provider mismatch $requested versus $actual", ({ requested, actual }) => {
  expect(isOnnxProviderInitializationError(new Error(`no available backend found. ERR: [${actual}] backend not found.`), requested)).toBe(false);
});
const rejectedCases: readonly { readonly requested: Accelerator | "cpu"; readonly message: string }[] = [
  { requested: "cuda", message: "Invalid ONNX model" },
  { requested: "cuda", message: "Cannot read tokenizer.json" },
  { requested: "cuda", message: "ONNX inference failed" },
  { requested: "cuda", message: "no available backend found. ERR: [cpu] backend not found." },
  { requested: "cpu", message: "no available backend found. ERR: [cpu] backend not found." },
  { requested: "cuda", message: "Unsupported device: \"cpu\". Should be one of: cuda." },
];
test.each(rejectedCases)("does not retry non-accelerator/provider/model errors: $message", ({ requested, message }) => {
  expect(isOnnxProviderInitializationError(new Error(message), requested)).toBe(false);
});
test("fails closed for non-Error rejection and hostile error accessors", () => {
  expect(isOnnxProviderInitializationError("no available backend found. ERR: [cuda]", "cuda")).toBe(false);
  const hostile = new Error("ignored");
  Object.defineProperty(hostile, "message", { get() { throw new Error("private detail"); } });
  expect(isOnnxProviderInitializationError(hostile, "cuda")).toBe(false);
});
