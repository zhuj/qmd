/**
 * onnx-sdk.test-d.ts - SDK embed-config and QMDStore embed-signal contracts.
 */
import { describe, test, expectTypeOf } from "vitest";
import type {
  CollectionConfig, ModelsConfig, OnnxModelSource,
  OnnxEmbeddingConfig, EmbeddingRecipe, QMDStore,
} from "../src/index.js";

describe("SDK embed types", () => {
  test("CollectionConfig and ModelsConfig embed agree", () => {
    expectTypeOf<
      NonNullable<CollectionConfig["models"]>["embed"]
    >().toEqualTypeOf<string | OnnxEmbeddingConfig | undefined>();
    expectTypeOf<ModelsConfig["embed"]>().toEqualTypeOf<
      string | OnnxEmbeddingConfig | undefined
    >();
  });

  test("OnnxEmbeddingConfig fields match source types", () => {
    expectTypeOf<OnnxEmbeddingConfig["source"]>().toEqualTypeOf<OnnxModelSource>();
    expectTypeOf<OnnxEmbeddingConfig["recipe"]>().toEqualTypeOf<EmbeddingRecipe>();
  });

  test("QMDStore embed signal and preserved model param", () => {
    expectTypeOf<
      NonNullable<Parameters<QMDStore["embed"]>[0]>["signal"]
    >().toEqualTypeOf<AbortSignal | undefined>();
    expectTypeOf<
      NonNullable<Parameters<QMDStore["embed"]>[0]>["model"]
    >().toEqualTypeOf<string | undefined>();
  });
});
