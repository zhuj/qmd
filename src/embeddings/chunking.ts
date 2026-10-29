import { isDeepStrictEqual } from "node:util";
import type { EmbeddingCapability, EmbeddingInput, EmbeddingRecipe, EmbeddingTokenMeasurement } from "./types.js";

export interface EmbeddingChunk { readonly text: string; readonly pos: number; readonly tokens: number }
export type TextSplitter = (text: string, maxChars: number) => Promise<readonly { readonly text: string; readonly pos: number }[]>;

function assertNever(value: never): never { throw new Error(`Unsupported embedding backend: ${String(value)}`); }
function countingContext(capability: EmbeddingCapability, recipe: EmbeddingRecipe) {
  switch (capability.config.backend) {
    case "gguf": throw new Error("ONNX chunking cannot use a GGUF producer");
    case "onnx": {
      if (!isDeepStrictEqual(recipe, capability.config.recipe)) throw new Error("Embedding recipe does not match configured producer");
      const operation = capability.measureTokens;
      if (!operation) throw new Error("ONNX producer token measurement is required");
      return {
        budget: Math.min(recipe.maxInputTokens, capability.config.maxBatchTokens),
        measure: async (input: EmbeddingInput, signal?: AbortSignal): Promise<EmbeddingTokenMeasurement> => {
          signal?.throwIfAborted();
          const result = await operation.call(capability, input, signal);
          signal?.throwIfAborted();
          return result;
        },
      };
    }
    default: return assertNever(capability.config);
  }
}
function scalarOffsets(text: string): readonly number[] {
  const offsets = [0]; let offset = 0;
  for (const scalar of text) { offset += scalar.length; offsets.push(offset); }
  return offsets;
}
async function fittingScalars(limit: number, fits: (count: number) => Promise<boolean>): Promise<number> {
  let low = 1, high = limit, best = 0;
  while (low <= high) {
    const middle = Math.floor((low + high) / 2);
    if (await fits(middle)) { best = middle; low = middle + 1; } else high = middle - 1;
  }
  // Token counts can decrease when additional characters form a merged token.
  if (best === 0 && limit > 0 && await fits(1)) return 1;
  return best;
}
function boundary(offsets: readonly number[], index: number): number {
  const offset = offsets[index];
  if (offset === undefined) throw new Error("Invalid Unicode scalar boundary");
  return offset;
}
export async function chunkEmbeddingDocument(
  capability: EmbeddingCapability, recipe: EmbeddingRecipe,
  body: string, title: string, split: TextSplitter, signal?: AbortSignal,
): Promise<readonly EmbeddingChunk[]> {
  signal?.throwIfAborted();
  const { budget, measure } = countingContext(capability, recipe);
  if (!body.isWellFormed()) throw new Error("Document must contain well-formed Unicode");
  if (!body) return Object.freeze([]);
  if ((await measure({ kind: "document", text: "", title }, signal)).total >= budget) {
    throw new Error("Embedding overhead leaves no positive body token budget");
  }
  const seeds = await split(body, Math.min(body.length, budget * 8));
  signal?.throwIfAborted();
  let covered = 0;
  for (const seed of seeds) {
    if (!Number.isSafeInteger(seed.pos) || seed.pos !== covered || !seed.text || !seed.text.isWellFormed()
      || body.slice(seed.pos, seed.pos + seed.text.length) !== seed.text) {
      throw new Error("Invalid splitter source interval or Unicode scalar boundary");
    }
    covered += seed.text.length;
  }
  if (covered !== body.length) throw new Error("Splitter intervals leave a source coverage gap or missing tail");
  const chunks: EmbeddingChunk[] = [];
  for (const seed of seeds) {
    let cursor = 0;
    while (cursor < seed.text.length) {
      signal?.throwIfAborted();
      // This bounds a candidate window, not an estimate of its actual token count.
      let window = seed.text.slice(cursor, cursor + budget * 8);
      if (!window.isWellFormed()) window = window.slice(0, -1);
      const offsets = scalarOffsets(window), measurements = new Map<number, EmbeddingTokenMeasurement>();
      const selected = await fittingScalars(offsets.length - 1, async count => {
        const result = await measure({ kind: "document", text: window.slice(0, boundary(offsets, count)), title }, signal);
        measurements.set(count, result); return result.total <= budget;
      });
      if (selected === 0) throw new Error("An indivisible document scalar exceeds the embedding token budget");
      const text = window.slice(0, boundary(offsets, selected)), measured = measurements.get(selected);
      if (!measured) throw new Error("Missing measured chunk boundary");
      chunks.push(Object.freeze({ text, pos: seed.pos + cursor, tokens: measured.total }));
      if (cursor + text.length === seed.text.length) break;
      const overlap = recipe.overlapTokens === 0 ? 0 : await fittingScalars(selected - 1, async count => {
        const suffix = text.slice(boundary(offsets, selected - count));
        return (await measure({ kind: "document", text: suffix, title }, signal)).text <= recipe.overlapTokens;
      });
      cursor += boundary(offsets, selected - overlap);
    }
  }
  signal?.throwIfAborted();
  return Object.freeze(chunks);
}
export async function assertEmbeddingQueryFits(
  capability: EmbeddingCapability, recipe: EmbeddingRecipe, query: string, signal?: AbortSignal,
): Promise<void> {
  signal?.throwIfAborted();
  const { budget, measure } = countingContext(capability, recipe);
  if ((await measure({ kind: "query", text: query }, signal)).total > budget) {
    throw new Error("Query exceeds embedding token budget; shorten it rather than truncating");
  }
}
