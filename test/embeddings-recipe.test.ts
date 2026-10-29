import { expect, test } from "vitest";
import { formatEmbeddingInput, poolEmbeddingOutput, validateEmbeddingVectors, type EmbeddingTokenizer, type EmbeddingModelOutput, type EmbeddingTensor } from "../src/embeddings/recipe.js";
import { makeOnnxConfig } from "./_helpers/embedding-config.js";
import type { EmbeddingRecipe } from "../src/embeddings/types.js";

function recipe(overrides: Partial<EmbeddingRecipe> = {}): EmbeddingRecipe {
  return { ...makeOnnxConfig().recipe, normalization: "none", ...overrides };
}
const hidden = { last_hidden_state: { dims: [2, 3, 2],
  data: new Float32Array([1, 0, 0, 2, 99, 99, 88, 88, 3, 0, 0, 4]) } };
const mask = [[1, 1, 0], [0, 1, 1]];
const values = (vectors: readonly Float32Array[]) => vectors.map((vector) => Array.from(vector));
const tokenizer: EmbeddingTokenizer = {
  encode: async (text) => Array.from(text, (character) => character.codePointAt(0) ?? 0),
  decode: async (tokens) => tokens.map((token) => String.fromCodePoint(token)).join(""),
};

test("last-token follows left and right padding masks", () => {
  expect(values(poolEmbeddingOutput(hidden, mask, recipe({ pooling: "last-token" })))).toEqual([[0, 2], [0, 4]]);
});
test("mean ignores padded values and preserves batch order", () => {
  expect(values(poolEmbeddingOutput(hidden, mask, recipe()))).toEqual([[0.5, 1], [1.5, 2]]);
});
test("CLS uses the declared unmasked token position", () => {
  expect(values(poolEmbeddingOutput(hidden, [[1, 1, 0], [1, 1, 1]], recipe({ pooling: "cls", clsTokenIndex: 1 }))))
    .toEqual([[0, 2], [3, 0]]);
});
test("native sentence output is not pooled a second time", () => {
  expect(values(poolEmbeddingOutput({ sentence_embedding: { dims: [2, 2], data: new Float32Array([1, 2, 3, 4]) } }, mask,
    recipe({ pooling: "sentence-output", sentenceOutput: "sentence_embedding" })))).toEqual([[1, 2], [3, 4]]);
});
test("L2 normalization is explicit, stable and does not mutate model tensors", () => {
  const output = { last_hidden_state: { dims: [1, 1, 2], data: new Float32Array([3, 4]) } };
  const vector = poolEmbeddingOutput(output, [[1]], recipe({ normalization: "l2" }))[0];
  expect(vector?.[0]).toBeCloseTo(0.6, 6);
  expect(vector?.[1]).toBeCloseTo(0.8, 6);
  expect(Array.from(output.last_hidden_state.data)).toEqual([3, 4]);
});
test("single-row and batched pooling agree exactly", () => {
  const first = { last_hidden_state: { dims: [1, 3, 2], data: hidden.last_hidden_state.data.slice(0, 6) } };
  expect(values(poolEmbeddingOutput(first, mask.slice(0, 1), recipe()))[0])
    .toEqual(values(poolEmbeddingOutput(hidden, mask, recipe()))[0]);
});
test.each([
  [], [[1, 1]], [[1, 1, 0]], [[1, 1, 0], [0, 0, 0]], [[1, 2, 0], [0, 1, 1]],
].map((invalid) => ({ invalid })))("rejects invalid attention mask %#", ({ invalid }) => {
  expect(() => poolEmbeddingOutput(hidden, invalid, recipe())).toThrow();
});
test.each([[2, 3], [2, 3, 3], [2, -1, 2], [2, 3, 2, 1], [2, 3, 1.5]])("rejects tensor shape %#", (...dims) => {
  expect(() => poolEmbeddingOutput({ last_hidden_state: { dims, data: hidden.last_hidden_state.data } }, mask, recipe())).toThrow();
});
test("rejects masked/out-of-range CLS and logits-only output", () => {
  expect(() => poolEmbeddingOutput(hidden, mask, recipe({ pooling: "cls", clsTokenIndex: 0 }))).toThrow();
  expect(() => poolEmbeddingOutput(hidden, mask, recipe({ pooling: "cls", clsTokenIndex: 3 }))).toThrow();
  expect(() => poolEmbeddingOutput({}, mask, recipe())).toThrow();
  const logits: EmbeddingModelOutput & { logits: EmbeddingTensor } = { logits: hidden.last_hidden_state };
  expect(() => poolEmbeddingOutput(logits, mask, recipe())).toThrow();
});
test.each([new Float32Array([NaN, 1]), new Float32Array([0, 0]), new Float32Array([Infinity, 0])])(
  "rejects invalid cosine vector %#", (vector) => expect(() => validateEmbeddingVectors([vector], 1, 2, undefined)).toThrow(),
);
test("requires exact count/native dimensions and rejects unsupported reduction", () => {
  const vector = new Float32Array([1, 2]);
  expect(() => validateEmbeddingVectors([vector], 2, 2, undefined)).toThrow();
  expect(() => validateEmbeddingVectors([vector], 1, 3, undefined)).toThrow();
  expect(() => validateEmbeddingVectors([vector], 1, 2, 1)).toThrow(/dimension|reduc/i);
  expect(() => validateEmbeddingVectors([vector], 1, 2, 2)).not.toThrow();
});
test("formats raw query/body once without truncating either", async () => {
  const config = recipe({ queryPrefix: "query: ", documentPrefix: "passage: ", titleBudgetTokens: 2 });
  expect(await formatEmbeddingInput({ kind: "query", text: "query: raw" }, config, tokenizer)).toBe("query: query: raw");
  expect(await formatEmbeddingInput({ kind: "document", title: "😀λx", text: "unmodified\nbody" }, config, tokenizer))
    .toBe("passage: 😀λ\nunmodified\nbody");
  expect(await formatEmbeddingInput({ kind: "document", title: "", text: "body" }, config, tokenizer)).toBe("passage: body");
});
test("backs off incomplete decoded Unicode at a title token boundary", async () => {
  const bytes: EmbeddingTokenizer = {
    encode: async (text) => [...Buffer.from(text)],
    decode: async (tokens) => Buffer.from(tokens).toString("utf8"),
  };
  expect(await formatEmbeddingInput({ kind: "document", title: "😀x", text: "body" }, recipe({ titleBudgetTokens: 2 }), bytes)).toBe("body");
});
test("backs off a well-formed decoded title that expands when retokenized", async () => {
  const expanding: EmbeddingTokenizer = {
    encode: async (text) => text === "ok" ? [1] : [1, 2, 3],
    decode: async (tokens) => tokens.length === 2 ? "expanded" : "ok",
  };
  expect(await formatEmbeddingInput({ kind: "document", title: "original", text: "body" },
    recipe({ titleBudgetTokens: 2 }), expanding)).toBe("ok\nbody");
});
