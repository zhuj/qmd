import { expect, test, describe } from "vitest";
import { planTokenBatches, type EmbeddingBatchLimits } from "../src/embeddings/onnx-batching.js";

function limits(maxItems: number, maxInputTokens: number, maxBatchTokens: number): EmbeddingBatchLimits {
  return Object.freeze({ maxItems, maxInputTokens, maxBatchTokens });
}

describe("planTokenBatches", () => {
  test("worked examples", () => {
    expect(planTokenBatches([3, 7], limits(2, 7, 14))).toEqual([[0, 1]]);
    expect(planTokenBatches([3, 7], limits(2, 7, 13))).toEqual([[0], [1]]);
  });

  test("maxInput rejects oversized item", () => {
    expect(() => planTokenBatches([7], limits(2, 6, 100))).toThrow(/maxInputTokens/);
  });

  test("lone length > maxBatchTokens rejects", () => {
    expect(() => planTokenBatches([4], limits(2, 10, 3))).toThrow(/maxBatchTokens/);
  });

  test("stable order and greedy with cap", () => {
    expect(planTokenBatches([2, 5, 1], limits(3, 10, 20))).toEqual([[0, 1, 2]]);
  });

  test("empty input -> frozen empty", () => {
    const r = planTokenBatches([], limits(2, 10, 20));
    expect(r).toEqual([]);
    expect(Object.isFrozen(r)).toBe(true);
  });

  test.each([
    { lengths: [3, 2, 4], limits: limits(2, 5, 10), want: [[0, 1], [2]] },
    { lengths: [1, 1, 1, 1], limits: limits(2, 2, 4), want: [[0, 1], [2, 3]] },
  ])("table batching $lengths", ({ lengths, limits: l, want }) => {
    expect(planTokenBatches(lengths, l)).toEqual(want);
  });

  test("invalid limits positive safe ints", () => {
    const bad0: unknown = { maxItems: 0, maxInputTokens: 10, maxBatchTokens: 10 };
    expect(() => planTokenBatches([1], bad0)).toThrow();
    const bad33: unknown = { maxItems: 33, maxInputTokens: 10, maxBatchTokens: 10 };
    expect(() => planTokenBatches([1], bad33)).toThrow();
    const bad0in: unknown = { maxItems: 2, maxInputTokens: 0, maxBatchTokens: 10 };
    expect(() => planTokenBatches([1], bad0in)).toThrow();
  });

  test("non-positive/non-integer lengths rejected", () => {
    expect(() => planTokenBatches([0], limits(2, 10, 10))).toThrow();
    expect(() => planTokenBatches([3.5], limits(2, 10, 10))).toThrow();
    expect(() => planTokenBatches([-1], limits(2, 10, 10))).toThrow();
  });

  test("deep freeze batches and outer result", () => {
    const r = planTokenBatches([2, 2], limits(2, 5, 4));
    expect(Object.isFrozen(r)).toBe(true);
    for (const b of r) expect(Object.isFrozen(b)).toBe(true);
  });

  test("no mutation of input lengths", () => {
    const input = [2, 3, 4];
    const copy = input.slice();
    planTokenBatches(input, limits(3, 10, 10));
    expect(input).toEqual(copy);
  });

  test("padded cost controls batching", () => {
    // same unpadded but different max -> split by padded cost
    expect(planTokenBatches([3, 3, 3], limits(3, 10, 9))).toEqual([[0, 1, 2]]); // max=3, 3*3=9
    expect(planTokenBatches([3, 3, 3], limits(3, 10, 8))).toEqual([[0, 1], [2]]); // 3*3=9>8
  });

  test("never drop/reorder/truncate", () => {
    const r = planTokenBatches([10, 1, 10, 1, 10], limits(2, 15, 30));
    const flat = r.flat();
    expect(flat).toEqual([0, 1, 2, 3, 4]);
  });
});
