import { expect, test } from "vitest";
import { createEmbeddingBudget } from "../src/embeddings/index-budget.js";
test("one deadline signal preserves caller abort reason and disposes the timer once", () => {
  const controller = new AbortController(); let callback: (() => void) | undefined, cancels = 0;
  const budget = createEmbeddingBudget(controller.signal, 100, (run, ms) => { expect(ms).toBe(100); callback = run; return () => { cancels++; }; });
  controller.abort("caller cancellation"); callback?.();
  expect(budget.signal.reason).toBe("caller cancellation"); expect(budget.expired()).toBe(true);
  budget.close(); budget.close(); expect(cancels).toBe(1);
});
test("deadline expiry aborts the operation without claiming native interruption", () => {
  let callback: (() => void) | undefined;
  const budget = createEmbeddingBudget(undefined, 50, run => { callback = run; return () => {}; });
  expect(budget.expired()).toBe(false); callback?.();
  expect(budget.expired()).toBe(true); expect(budget.signal.aborted).toBe(true);
  expect(budget.signal.reason).toEqual(new Error("Embedding time limit exceeded")); budget.close();
});
test.each([{ milliseconds: 0 }, { milliseconds: -1 }])("disabled deadlines allocate no timer", ({ milliseconds }) => {
  const budget = createEmbeddingBudget(undefined, milliseconds, () => { throw new Error("unexpected timer"); });
  expect(budget.signal.aborted).toBe(false); expect(budget.expired()).toBe(false); budget.close();
});
test.each([{ milliseconds: NaN }, { milliseconds: Infinity }, { milliseconds: 2147483648 }])("invalid deadlines reject before scheduling", ({ milliseconds }) => {
  expect(() => createEmbeddingBudget(undefined, milliseconds, () => { throw new Error("unexpected timer"); })).toThrow(/duration|limit/i);
});
