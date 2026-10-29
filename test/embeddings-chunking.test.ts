import { expect, test } from "vitest";
import { formatEmbeddingInput, type EmbeddingTokenizer } from "../src/embeddings/recipe.js";
import { resolveEmbeddingConfig } from "../src/embeddings/config.js";
import { chunkEmbeddingDocument, assertEmbeddingQueryFits, type TextSplitter } from "../src/embeddings/chunking.js";
import { makeOnnxConfig } from "./_helpers/embedding-config.js";
import type { EmbeddingCapability, EmbeddingRecipe } from "../src/embeddings/types.js";

function fixture(overrides: Partial<EmbeddingRecipe> = {}, maxBatchTokens = 64): EmbeddingCapability {
  const base = makeOnnxConfig();
  const config = resolveEmbeddingConfig({ ...base, recipe: { ...base.recipe, queryPrefix: "Q:", documentPrefix: "D:",
    maxInputTokens: 12, titleBudgetTokens: 2, overlapTokens: 1, ...overrides }, maxBatchTokens },
  { defaultGgufModel: "legacy", hostParallelism: 8 });
  if (config.backend !== "onnx") throw new Error("Wrong fixture backend");
  const tokenizer: EmbeddingTokenizer = {
    encode: async (text, special) => [...(special ? [0] : []), ...Array.from(text, c => c.codePointAt(0) ?? 0), ...(special ? [0] : [])],
    decode: async ids => String.fromCodePoint(...ids),
  };
  return { config, modelLabel: "fake-scalars", provenance: undefined,
    prepare: async () => { throw new Error("Preparation forbidden in pure counting tests"); },
    encode: async () => { throw new Error("Literal encode cannot reproduce producer formatting"); },
    decode: async () => { throw new Error("Source reconstruction by decode forbidden"); },
    embed: async () => { throw new Error("Inference forbidden"); }, close: async () => {},
    measureTokens: async (input, signal) => {
      signal?.throwIfAborted();
      return { text: (await tokenizer.encode(input.text, false)).length,
        total: (await tokenizer.encode(await formatEmbeddingInput(input, config.recipe, tokenizer), true)).length };
    } };
}
function recipeOf(cap: EmbeddingCapability): EmbeddingRecipe {
  if (cap.config.backend !== "onnx") throw new Error("Expected ONNX fixture");
  return cap.config.recipe;
}
const whole: TextSplitter = async text => [{ text, pos: 0 }];
test("chunks preserve every original interval and Unicode tail under exact producer formatting", async () => {
  const cap = fixture(), recipe = recipeOf(cap), body = "alpha 😀 中文 e\u0301  ".repeat(8) + "TAIL終";
  const chunks = await chunkEmbeddingDocument(cap, recipe, body, "long title", whole);
  expect(chunks.length).toBeGreaterThan(1); expect(Object.isFrozen(chunks)).toBe(true);
  let end = 0, previous = -1;
  for (const chunk of chunks) {
    expect(chunk.pos).toBeGreaterThan(previous); expect(chunk.pos).toBeLessThanOrEqual(end);
    expect(chunk.text).toBe(body.slice(chunk.pos, chunk.pos + chunk.text.length));
    expect(chunk.text.isWellFormed()).toBe(true); expect(Object.isFrozen(chunk)).toBe(true);
    expect(chunk.tokens).toBe(Array.from(chunk.text).length + 7);
    expect(chunk.tokens).toBeLessThanOrEqual(12); end = Math.max(end, chunk.pos + chunk.text.length); previous = chunk.pos;
  }
  expect(end).toBe(body.length); expect(chunks.at(-1)?.text.endsWith("終")).toBe(true);
});
test("raw token overlap respects budget and the final tail is not re-emitted", async () => {
  const cap = fixture({ maxInputTokens: 10, titleBudgetTokens: 0, overlapTokens: 2 });
  const chunks = await chunkEmbeddingDocument(cap, recipeOf(cap), "abcdefghijkl", "", whole);
  expect(chunks.map(c => [c.pos, c.text])).toEqual([[0, "abcdef"], [4, "efghij"], [8, "ijkl"]]);
});
test("batch ceiling tightens the real formatted budget", async () => {
  const cap = fixture({ maxInputTokens: 20, titleBudgetTokens: 0, overlapTokens: 0 }, 8);
  const chunks = await chunkEmbeddingDocument(cap, recipeOf(cap), "abcdef", "", whole);
  expect(chunks.map(c => c.text)).toEqual(["abcd", "ef"]); expect(chunks.map(c => c.tokens)).toEqual([8, 6]);
});
test("source-aligned splitter boundaries are honored rather than discarded", async () => {
  const cap = fixture({ maxInputTokens: 10, titleBudgetTokens: 0, overlapTokens: 0 });
  const chunks = await chunkEmbeddingDocument(cap, recipeOf(cap), "abcdefghij", "", async () => [
    { text: "abcde", pos: 0 }, { text: "fghij", pos: 5 },
  ]);
  expect(chunks.map(c => [c.pos, c.text])).toEqual([[0, "abcde"], [5, "fghij"]]);
});
test("long-source candidate measurements remain bounded without estimating actual token cost", async () => {
  const cap = fixture({ maxInputTokens: 8, titleBudgetTokens: 0, overlapTokens: 0 });
  const measure = cap.measureTokens;
  if (!measure) throw new Error("Fixture measurement missing");
  const lengths: number[] = [];
  const observed: EmbeddingCapability = { ...cap, measureTokens: async (input, signal) => {
    lengths.push(input.text.length); return measure.call(cap, input, signal);
  } };
  const body = "a".repeat(200);
  const chunks = await chunkEmbeddingDocument(observed, recipeOf(cap), body, "", whole);
  expect(Math.max(...lengths)).toBeLessThanOrEqual(64);
  expect(chunks.map(c => c.text).join("")).toBe(body);
});
test("query exact limit passes and one scalar over rejects without truncation", async () => {
  const cap = fixture({ maxInputTokens: 6, titleBudgetTokens: 0, overlapTokens: 0 });
  await assertEmbeddingQueryFits(cap, recipeOf(cap), "ab");
  await expect(assertEmbeddingQueryFits(cap, recipeOf(cap), "abc")).rejects.toThrow(/budget|overflow|limit/i);
});
test("configured prefixes and normalization participate in recipe equality", async () => {
  const cap = fixture(), recipe = recipeOf(cap);
  await expect(chunkEmbeddingDocument(cap, { ...recipe, documentPrefix: "other" }, "abc", "", whole)).rejects.toThrow(/recipe/i);
  await expect(assertEmbeddingQueryFits(cap, { ...recipe, normalization: "none" }, "x")).rejects.toThrow(/recipe/i);
});
test("GGUF is rejected even if it happens to expose measurement", async () => {
  const cap = fixture(), recipe = recipeOf(cap);
  await expect(chunkEmbeddingDocument({ ...cap, config: { backend: "gguf", model: "legacy" } }, recipe, "x", "", whole)).rejects.toThrow(/onnx|gguf/i);
});
test("missing measurement cannot fall back to a different tokenizer", async () => {
  const cap = fixture(), { measureTokens: _omitted, ...without } = cap;
  await expect(chunkEmbeddingDocument(without, recipeOf(cap), "x", "", whole)).rejects.toThrow(/measure/i);
});
test("empty body returns frozen empty chunks", async () => {
  const cap = fixture(); expect(await chunkEmbeddingDocument(cap, recipeOf(cap), "", "", whole)).toEqual([]);
});
test.each([
  { split: async () => [] }, { split: async () => [{ text: "", pos: 0 }] },
  { split: async () => [{ text: "a", pos: 0 }, { text: "c", pos: 2 }] },
  { split: async () => [{ text: "wrong", pos: 0 }] }, { split: async () => [{ text: "ab", pos: 0 }] },
])("malformed splitter cannot silently lose body coverage", async ({ split }) => {
  const cap = fixture(); await expect(chunkEmbeddingDocument(cap, recipeOf(cap), "abc", "", split)).rejects.toThrow(/split|seed|gap|tail|cover/i);
});
test("splitter boundaries cannot split a surrogate pair", async () => {
  const cap = fixture();
  await expect(chunkEmbeddingDocument(cap, recipeOf(cap), "😀", "", async text => [
    { text: text.slice(0, 1), pos: 0 }, { text: text.slice(1), pos: 1 },
  ])).rejects.toThrow(/unicode|scalar|split/i);
});
test("overhead and an indivisible over-budget scalar fail actionably", async () => {
  const cap = fixture({ maxInputTokens: 4, titleBudgetTokens: 0, overlapTokens: 0 });
  await expect(chunkEmbeddingDocument(cap, recipeOf(cap), "x", "", whole)).rejects.toThrow(/overhead|body|budget/i);
  const ordinary = fixture({ maxInputTokens: 8, titleBudgetTokens: 0, overlapTokens: 0 });
  const measured: EmbeddingCapability = { ...ordinary, measureTokens: async input => ({ text: input.text.length, total: input.text ? 99 : 2 }) };
  await expect(chunkEmbeddingDocument(measured, recipeOf(ordinary), "x", "", whole)).rejects.toThrow(/scalar|budget|overflow/i);
});
test("non-Error cancellation retains the exact reason before and during counting", async () => {
  const cap = fixture(), controller = new AbortController(), reason = "stop-exactly"; controller.abort(reason);
  await expect(chunkEmbeddingDocument(cap, recipeOf(cap), "abc", "", whole, controller.signal)).rejects.toBe(reason);
  const later = new AbortController();
  const measured: EmbeddingCapability = { ...cap, measureTokens: async () => { later.abort(reason); return { text: 0, total: 4 }; } };
  await expect(chunkEmbeddingDocument(measured, recipeOf(cap), "abc", "", whole, later.signal)).rejects.toBe(reason);
});
