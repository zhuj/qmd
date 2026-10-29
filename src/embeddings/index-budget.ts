export interface EmbeddingBudget {
  readonly signal: AbortSignal;
  expired(): boolean;
  close(): void;
}
export type EmbeddingDeadlineClock = (callback: () => void, milliseconds: number) => () => void;
const scheduleDeadline: EmbeddingDeadlineClock = (callback, milliseconds) => {
  const timer = setTimeout(callback, milliseconds); timer.unref(); return () => clearTimeout(timer);
};
export function createEmbeddingBudget(signal: AbortSignal | undefined, milliseconds: number, clock: EmbeddingDeadlineClock = scheduleDeadline): EmbeddingBudget {
  if (!Number.isFinite(milliseconds) || milliseconds > 2147483647) throw new Error("Embedding duration is outside the supported timer limit");
  const controller = new AbortController();
  let expired = false, closed = false;
  const cancel = milliseconds > 0 ? clock(() => {
    if (closed) return;
    expired = true; controller.abort(new Error("Embedding time limit exceeded"));
  }, milliseconds) : undefined;
  return Object.freeze({
    signal: signal ? AbortSignal.any([signal, controller.signal]) : controller.signal,
    expired: () => expired,
    close: () => { if (!closed) { closed = true; cancel?.(); } },
  });
}
