import { expect, test, describe } from "vitest";
import { readInt64Tensor, readFloatTensor } from "../src/embeddings/onnx-native-values.js";

describe("readInt64Tensor structural validators", () => {
  const ok2 = { type: "int64", location: "cpu", dims: [2, 3], data: new BigInt64Array([1n, 2n, 3n, 4n, 5n, 6n]) };

  test.each([
    { label: "rank2 positive dims and matching BigInt64Array", input: ok2 },
  ])("accepts $label", ({ input }) => {
    const dataBefore = new BigInt64Array(input.data);
    const result = readInt64Tensor(input);
    expect(result.native).toBe(input);
    expect(result.dims).toEqual(input.dims);
    expect(result.data).toBe(input.data);
    expect(input.data).toEqual(dataBefore);
  });

  test("does not modify input object (int64)", () => {
    const input = { ...ok2, dims: [...ok2.dims] };
    const dataBefore = new BigInt64Array(input.data);
    const metaBefore = JSON.stringify({ dims: input.dims, type: input.type, location: input.location, len: input.data.length });
    readInt64Tensor(input);
    expect(JSON.stringify({ dims: input.dims, type: input.type, location: input.location, len: input.data.length })).toBe(metaBefore);
    expect(input.data).toEqual(dataBefore);
  });

  const badInt64Cases = [
    { label: "wrong dtype float32", input: { type: "float32", location: "cpu", dims: [2, 2], data: new BigInt64Array([1n, 2n, 3n, 4n]) } },
    { label: "wrong location cuda", input: { type: "int64", location: "cuda", dims: [2, 2], data: new BigInt64Array([1n, 2n, 3n, 4n]) } },
    { label: "Float32Array instead of BigInt64Array", input: { type: "int64", location: "cpu", dims: [2, 2], data: new Float32Array([1, 2, 3, 4]) } },
    { label: "rank1 dims rejected", input: { type: "int64", location: "cpu", dims: [3], data: new BigInt64Array([1n, 2n, 3n]) } },
    { label: "rank4 dims rejected", input: { type: "int64", location: "cpu", dims: [1, 1, 1, 1], data: new BigInt64Array([1n]) } },
    { label: "zero dimension rejected", input: { type: "int64", location: "cpu", dims: [2, 0], data: new BigInt64Array([1n]) } },
    { label: "negative dimension rejected", input: { type: "int64", location: "cpu", dims: [2, -1], data: new BigInt64Array([1n, 2n]) } },
    { label: "cardinality too high", input: { type: "int64", location: "cpu", dims: [2, 2], data: new BigInt64Array([1n, 2n, 3n, 4n, 5n]) } },
    { label: "cardinality too low", input: { type: "int64", location: "cpu", dims: [2, 2], data: new BigInt64Array([1n, 2n]) } },
  ];
  test.each(badInt64Cases)("rejects $label", ({ input }) => {
    expect(() => readInt64Tensor(input)).toThrow();
  });
});

describe("readFloatTensor structural validators", () => {
  const okRank2 = { type: "float32", location: "cpu", dims: [2, 3], data: new Float32Array([1, 2, 3, 4, 5, 6]) };
  const okRank3 = { type: "float32", location: "cpu", dims: [1, 2, 2], data: new Float32Array([10, 20, 30, 40]) };

  test.each([
    { label: "rank2 positive dims and matching Float32Array", input: okRank2 },
    { label: "rank3 positive dims and matching Float32Array", input: okRank3 },
  ])("accepts $label", ({ input }) => {
    const dataBefore = new Float32Array(input.data);
    const result = readFloatTensor(input);
    expect(result.dims).toEqual(input.dims);
    expect(result.data).toBe(input.data);
    expect(input.data).toEqual(dataBefore);
  });

  test("does not modify input object (float)", () => {
    const input = { ...okRank2, dims: [...okRank2.dims] };
    const dataBefore = new Float32Array(input.data);
    const metaBefore = JSON.stringify({ dims: input.dims, type: input.type, location: input.location, len: input.data.length });
    readFloatTensor(input);
    expect(JSON.stringify({ dims: input.dims, type: input.type, location: input.location, len: input.data.length })).toBe(metaBefore);
    expect(input.data).toEqual(dataBefore);
  });

  const badFloatCases = [
    { label: "wrong dtype int64", input: { type: "int64", location: "cpu", dims: [2, 2], data: new Float32Array([1, 2, 3, 4]) } },
    { label: "wrong location gpu", input: { type: "float32", location: "gpu", dims: [2, 2], data: new Float32Array([1, 2, 3, 4]) } },
    { label: "BigInt64Array instead of Float32Array", input: { type: "float32", location: "cpu", dims: [2, 2], data: new BigInt64Array([1n, 2n, 3n, 4n]) } },
    { label: "rank1 dims rejected", input: { type: "float32", location: "cpu", dims: [3], data: new Float32Array([1, 2, 3]) } },
    { label: "rank4 dims rejected", input: { type: "float32", location: "cpu", dims: [1, 1, 1, 1], data: new Float32Array([1]) } },
    { label: "zero dimension rejected", input: { type: "float32", location: "cpu", dims: [2, 0], data: new Float32Array([1]) } },
    { label: "negative dimension rejected", input: { type: "float32", location: "cpu", dims: [2, -3], data: new Float32Array([1, 2, 3]) } },
    { label: "cardinality too high", input: { type: "float32", location: "cpu", dims: [2, 2], data: new Float32Array([1, 2, 3, 4, 5]) } },
    { label: "cardinality too low", input: { type: "float32", location: "cpu", dims: [2, 2], data: new Float32Array([1, 2]) } },
  ];
  test.each(badFloatCases)("rejects $label", ({ input }) => {
    expect(() => readFloatTensor(input)).toThrow();
  });

  test("rejects non-objects for float32 (unknown raw)", () => {
    expect(() => readFloatTensor(42)).toThrow();
    expect(() => readFloatTensor(null)).toThrow();
  });
  test("rejects non-objects for int64 (unknown raw)", () => {
    expect(() => readInt64Tensor(42)).toThrow();
    expect(() => readInt64Tensor(null)).toThrow();
  });
});
