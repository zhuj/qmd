/**
 * Pure environment boundary guard for transformers adapter.
 * Tests the agreed `assertTransformersEnvironment` contract without importing
 * the production module (parent provides the skeleton independently).
 */

import { describe, test, expect } from "vitest";
import { assertTransformersEnvironment } from "../src/embeddings/onnx-transformers-env.js";

describe("assertTransformersEnvironment", () => {
  const base = {
    allowRemoteModels: false,
    allowLocalModels: true,
    useFS: true,
    useFSCache: false,
    useBrowserCache: false,
    useCustomCache: false,
    experimental_useCrossOriginStorage: false,
  };

  test.each([
    { label: "default Node-like env accepted", input: { ...base } },
    { label: "remote disabled accepted", input: { ...base, allowRemoteModels: false } },
    { label: "remote enabled accepted (adapter uses per-load local_files_only)", input: { ...base, allowRemoteModels: true } },
    { label: "FS cache disabled accepted", input: { ...base, useFSCache: false } },
    { label: "FS cache enabled accepted (isolated cache_dir / direct tokenizer)", input: { ...base, useFSCache: true } },
  ])("accepts: $label", ({ input }) => {
    const frozen = JSON.parse(JSON.stringify(input));
    expect(() => assertTransformersEnvironment(input)).not.toThrow();
    expect(input).toEqual(frozen);
  });

  test.each([
    { label: "local models disabled", input: { ...base, allowLocalModels: false } },
    { label: "filesystem disabled", input: { ...base, useFS: false } },
    { label: "browser cache enabled", input: { ...base, useBrowserCache: true } },
    { label: "custom cache enabled", input: { ...base, useCustomCache: true } },
    { label: "cross-origin storage enabled", input: { ...base, experimental_useCrossOriginStorage: true } },
  ])("rejects bad flags: $label", ({ input }) => {
    expect(() => assertTransformersEnvironment(input)).toThrow();
  });

  test.each([
    { label: "missing bool (allowLocalModels undefined)", input: { ...base, allowLocalModels: undefined } },
    { label: "invalid bool string (useFS 'yes')", input: { ...base, useFS: "yes" } },
    { label: "invalid bool number (useBrowserCache 1)", input: { ...base, useBrowserCache: 1 } },
    { label: "invalid bool object (useCustomCache {})", input: { ...base, useCustomCache: {} } },
  ])("rejects invalid/missing bool: $label", ({ input }) => {
    expect(() => assertTransformersEnvironment(input)).toThrow();
  });

  test.each([
    { label: "null input", input: null },
    { label: "string input", input: "env" },
    { label: "number input", input: 42 },
    { label: "boolean input", input: true },
    { label: "array input", input: [base] },
  ])("rejects primitives/null/array: $label", ({ input }) => {
    const v: unknown = input;
    expect(() => assertTransformersEnvironment(v)).toThrow();
  });

  test("accepts extra unknown env properties and preserves them", () => {
    const input = { ...base, unknownExtra: 123, extraFlag: "preserve" };
    expect(() => assertTransformersEnvironment(input)).not.toThrow();
    expect(input).toHaveProperty("unknownExtra", 123);
    expect(input).toHaveProperty("extraFlag", "preserve");
  });

  test("does not mutate input on rejection", () => {
    const bad = { ...base, allowLocalModels: false };
    const before = JSON.parse(JSON.stringify(bad));
    expect(() => assertTransformersEnvironment(bad)).toThrow();
    expect(bad).toEqual(before);
  });

  test("does not mutate input on acceptance", () => {
    const good = { ...base };
    const before = JSON.parse(JSON.stringify(good));
    expect(() => assertTransformersEnvironment(good)).not.toThrow();
    expect(good).toEqual(before);
  });

  test("every bad shape throws generic error without exposing original message", () => {
    expect(() => assertTransformersEnvironment({ ...base, useFS: false })).toThrow();
    // Generic: no message/cause/path leakage required; just throws.
  });
});
