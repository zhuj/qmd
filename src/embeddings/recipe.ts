import type { EmbeddingInput, EmbeddingRecipe, EmbeddingTokenMeasurement } from "./types.js";
export interface EmbeddingTokenizer {
  encode(text: string, addSpecialTokens: boolean): Promise<readonly number[]>;
  decode(tokens: readonly number[]): Promise<string>;
}
export interface EmbeddingTensor { readonly data: Float32Array; readonly dims: readonly number[] }
export interface EmbeddingModelOutput {
  readonly last_hidden_state?: EmbeddingTensor;
  readonly sentence_embedding?: EmbeddingTensor;
}
function assertNever(value: never): never { throw new Error(`Unsupported embedding variant: ${String(value)}`); }
export async function formatEmbeddingInput(input: EmbeddingInput, recipe: EmbeddingRecipe, tokenizer: EmbeddingTokenizer): Promise<string> {
  switch (input.kind) {
    case "query": return recipe.queryPrefix + input.text;
    case "document": {
      let title = input.title;
      const tokens = await tokenizer.encode(title, false);
      if (tokens.length > recipe.titleBudgetTokens) {
        title = "";
        for (let length = recipe.titleBudgetTokens; length > 0; length--) {
          const candidate = await tokenizer.decode(tokens.slice(0, length));
          // A BPE token boundary can split a UTF-8 character; only the title may back off.
          if (candidate.isWellFormed() && !candidate.includes("\uFFFD")
            && (await tokenizer.encode(candidate, false)).length <= recipe.titleBudgetTokens) {
            title = candidate;
            break;
          }
        }
      }
      return recipe.documentPrefix + (title ? `${title}\n` : "") + input.text;
    }
    default: return assertNever(input);
  }
}
export async function measureEmbeddingInput(input: EmbeddingInput, recipe: EmbeddingRecipe, tokenizer: EmbeddingTokenizer): Promise<EmbeddingTokenMeasurement> {
  const formatted = await formatEmbeddingInput(input, recipe, tokenizer);
  const text = (await tokenizer.encode(input.text, false)).length;
  const total = (await tokenizer.encode(formatted, true)).length;
  return Object.freeze({ text, total });
}
export function poolEmbeddingOutput(output: EmbeddingModelOutput, mask: readonly (readonly number[])[], recipe: EmbeddingRecipe): readonly Float32Array[] {
  if (Object.keys(output).some((key) => key !== "last_hidden_state" && key !== "sentence_embedding")) {
    throw new Error("Unsupported embedding model output; logits are not embeddings");
  }
  const batch = mask.length;
  const sequence = mask[0]?.length ?? 0;
  if (!batch || !sequence || mask.some((row) => row.length !== sequence
    || Array.from(row).some((value) => value !== 0 && value !== 1) || !row.includes(1))) throw new Error("Invalid embedding attention mask");
  let tensor: EmbeddingTensor | undefined;
  let rank: number;
  switch (recipe.pooling) {
    case "sentence-output": tensor = output.sentence_embedding; rank = 2; break;
    case "mean": case "cls": case "last-token": tensor = output.last_hidden_state; rank = 3; break;
    default: return assertNever(recipe.pooling);
  }
  const dimensions = tensor?.dims[rank - 1];
  if (!tensor || dimensions === undefined || !(tensor.data instanceof Float32Array) || tensor.dims.length !== rank
    || Array.from(tensor.dims).some((dimension) => !Number.isSafeInteger(dimension) || dimension <= 0)
    || tensor.dims[0] !== batch || (rank === 3 && tensor.dims[1] !== sequence)
    || !Number.isSafeInteger(tensor.dims.reduce((size, dimension) => size * dimension, 1))
    || tensor.dims.reduce((size, dimension) => size * dimension, 1) !== tensor.data.length) {
    throw new Error("Invalid embedding tensor shape or selected output");
  }
  const vectors = mask.map((row, batchIndex) => {
    const vector = new Float32Array(dimensions);
    const start = batchIndex * (rank === 2 ? dimensions : sequence * dimensions);
    switch (recipe.pooling) {
      case "sentence-output": vector.set(tensor.data.subarray(start, start + dimensions)); break;
      case "mean": {
        const count = row.reduce((sum, value) => sum + value, 0);
        for (let dimension = 0; dimension < dimensions; dimension++) {
          let sum = 0;
          for (let token = 0; token < sequence; token++) {
            if (row[token] !== 1) continue;
            const value = tensor.data[start + token * dimensions + dimension];
            if (value === undefined) throw new Error("Embedding tensor scalar is outside its declared shape");
            sum += value;
          }
          vector[dimension] = sum / count;
        }
        break;
      }
      case "cls": {
        const index = recipe.clsTokenIndex;
        if (index === undefined || !Number.isSafeInteger(index) || index < 0 || index >= sequence || row[index] !== 1) {
          throw new Error("CLS position is outside the unmasked sequence");
        }
        vector.set(tensor.data.subarray(start + index * dimensions, start + (index + 1) * dimensions));
        break;
      }
      case "last-token": {
        const index = row.lastIndexOf(1);
        vector.set(tensor.data.subarray(start + index * dimensions, start + (index + 1) * dimensions));
        break;
      }
      default: return assertNever(recipe.pooling);
    }
    return vector;
  });
  validateEmbeddingVectors(vectors, batch, dimensions, undefined);
  switch (recipe.normalization) {
    case "none": break;
    case "l2":
      for (const vector of vectors) {
        const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0));
        vector.forEach((value, index) => { vector[index] = value / norm; });
      }
      break;
    default: return assertNever(recipe.normalization);
  }
  return Object.freeze(vectors);
}
export function validateEmbeddingVectors(vectors: readonly Float32Array[], count: number, nativeDimensions: number, requestedDimensions: number | undefined): void {
  if (!Number.isSafeInteger(count) || count < 0 || vectors.length !== count) throw new Error("Embedding output count mismatch");
  if (!Number.isSafeInteger(nativeDimensions) || nativeDimensions <= 0
    || (requestedDimensions !== undefined && requestedDimensions !== nativeDimensions)) {
    throw new Error("Unsupported embedding dimensions; reduction requires a documented supported model profile");
  }
  for (const vector of vectors) {
    if (!(vector instanceof Float32Array) || vector.length !== nativeDimensions
      || vector.some((value) => !Number.isFinite(value)) || !vector.some((value) => value !== 0)) {
      throw new Error("Embedding vectors must have the exact dimensions, finite float32 values and positive norm");
    }
  }
}
