import { z } from "zod";
import type { EmbeddingTensor } from "./recipe.js";

interface NativeMembers { readonly [key: string]: unknown }
type NativeFunction = (this: unknown, ...args: unknown[]) => unknown;
function hasNativeMembers(value: unknown): value is NativeMembers {
  return value !== null && (typeof value === "object" || typeof value === "function");
}
function isNativeFunction(value: unknown): value is NativeFunction { return typeof value === "function"; }
export function nativeFunction(value: unknown): NativeFunction {
  if (!isNativeFunction(value)) throw new Error("Unsupported ONNX session contract");
  return value;
}
export function nativeObject(value: unknown): NativeMembers {
  if (!hasNativeMembers(value) || typeof value !== "object") throw new Error("Unsupported ONNX session contract");
  return value;
}
export function nativeProperty(value: unknown, key: string): unknown {
  if (!hasNativeMembers(value)) throw new Error("Unsupported ONNX session contract");
  return value[key];
}
export function nativeMethod(value: unknown, key: string): (...args: unknown[]) => unknown {
  const method = nativeFunction(nativeProperty(value, key));
  return (...args): unknown => method.call(value, ...args);
}
export class NativeTensorScope {
  private readonly owners = new Map<object, () => unknown>();
  private releasing: Promise<void> | undefined;
  transformer(value: unknown): object {
    const native = nativeObject(nativeProperty(value, "ort_tensor"));
    if (!this.owners.has(native)) this.owners.set(native, nativeMethod(value, "dispose"));
    return native;
  }
  outputs(value: unknown): void {
    let failed = false;
    for (const tensor of Object.values(nativeObject(value))) {
      try {
        const native = nativeObject(tensor);
        if (!this.owners.has(native)) this.owners.set(native, nativeMethod(native, "dispose"));
      } catch { failed = true; }
    }
    if (failed) throw new Error("Unsupported ONNX output tensor");
  }
  inputs(value: unknown): void {
    let failed = false;
    for (const tensor of Object.values(nativeObject(value))) {
      try { this.transformer(tensor); } catch { failed = true; }
    }
    if (failed) throw new Error("Unsupported ONNX input tensor");
  }
  release(): Promise<void> {
    return this.releasing ??= Promise.resolve().then(async () => {
      let failed = false;
      for (const dispose of this.owners.values()) { try { await dispose(); } catch { failed = true; } }
      if (failed) throw new Error("ONNX tensor cleanup failed");
    });
  }
}
const dimensions = z.array(z.number().int().positive()).min(2).max(3);
export interface NativeInt64Tensor {
  readonly native: NativeMembers;
  readonly dims: readonly number[];
  readonly data: BigInt64Array;
}
export function readInt64Tensor(value: unknown): NativeInt64Tensor {
  const native = nativeObject(value);
  if (nativeProperty(native, "type") !== "int64" || nativeProperty(native, "location") !== "cpu") throw new Error("Unsupported ONNX input tensor");
  const dims = dimensions.length(2).parse(nativeProperty(native, "dims"));
  const data = nativeProperty(native, "data");
  if (!(data instanceof BigInt64Array) || dims.reduce((a, b) => a * b, 1) !== data.length) throw new Error("Invalid ONNX input shape");
  return { native, dims, data };
}
export function readFloatTensor(value: unknown): EmbeddingTensor {
  const native = nativeObject(value);
  if (nativeProperty(native, "type") !== "float32" || nativeProperty(native, "location") !== "cpu") throw new Error("Unsupported ONNX output tensor");
  const dims = dimensions.parse(nativeProperty(native, "dims")), data = nativeProperty(native, "data");
  if (!(data instanceof Float32Array) || dims.reduce((a, b) => a * b, 1) !== data.length) throw new Error("Invalid ONNX output shape");
  return { dims, data };
}
