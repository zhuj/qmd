import { z } from "zod";

export type EmbeddingBatchLimits = Readonly<{
  readonly maxItems: number;
  readonly maxInputTokens: number;
  readonly maxBatchTokens: number;
}>;
const limitsSchema = z.strictObject({
  maxItems: z.number().int().min(1).max(32),
  maxInputTokens: z.number().int().min(1).max(65536),
  maxBatchTokens: z.number().int().min(1).max(65536),
});

export function planTokenBatches(
  lengths: readonly number[],
  rawLimits: unknown,
): readonly (readonly number[])[] {
  const parsedLimits = limitsSchema.safeParse(rawLimits);
  if (!parsedLimits.success) throw new RangeError("Embedding batch limits are invalid");
  const { maxItems, maxInputTokens, maxBatchTokens } = parsedLimits.data;

  for (const [index, length] of lengths.entries()) {
    if (!Number.isSafeInteger(length) || length <= 0) {
      throw new RangeError(`lengths[${index}] must be a positive integer`);
    }
    if (length > maxInputTokens) {
      throw new RangeError(`lengths[${index}] exceeds maxInputTokens`);
    }
    if (length > maxBatchTokens) {
      throw new RangeError(`lengths[${index}] exceeds maxBatchTokens`);
    }
  }

  const batches: number[][] = [];
  let cur: number[] = [];
  let sum = 0;
  let maxInCur = 0;

  function flush() {
    if (cur.length > 0) {
      batches.push(cur);
      cur = [];
      sum = 0;
      maxInCur = 0;
    }
  }

  for (const [idx, len] of lengths.entries()) {
    const nextCount = cur.length + 1;
    const unpadded = sum + len;
    const newMax = cur.length === 0 ? len : Math.max(maxInCur, len);
    const padded = newMax * nextCount;
    if (cur.length > 0 && (nextCount > maxItems || unpadded > maxBatchTokens || padded > maxBatchTokens)) {
      flush();
    }
    // After possible flush, cur empty
    if (cur.length === 0) {
      cur = [idx];
      sum = len;
      maxInCur = len;
    } else {
      cur.push(idx);
      sum += len;
      maxInCur = Math.max(maxInCur, len);
    }
  }
  flush();

  const frozenBatches: readonly (readonly number[])[] = batches.map(b => Object.freeze(b.slice()));
  const result = Object.freeze(frozenBatches);
  // Deep-freeze batches content already frozen; freeze outer
  return result;
}
