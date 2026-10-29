import type { EmbeddingCapability } from "./types.js";

export function leaseEmbeddingCapability(raw: EmbeddingCapability): EmbeddingCapability {
  const measureTokens = raw.measureTokens;
  let active = 0;
  let closing: Promise<void> | undefined;
  let drained: (() => void) | undefined;

  async function run<T>(operation: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    if (closing) throw new Error("Embedding capability is closing or closed");
    signal?.throwIfAborted();
    active++;
    try {
      const result = await operation();
      // A native run may be uninterruptible: keep its lease but withhold cancelled results.
      signal?.throwIfAborted();
      return result;
    } finally {
      active--;
      if (active === 0) {
        drained?.();
        drained = undefined;
      }
    }
  }

  return Object.freeze({
    config: raw.config,
    modelLabel: raw.modelLabel,
    get provenance() { return raw.provenance; },
    prepare: (signal?: AbortSignal) => run(() => raw.prepare(signal), signal),
    encode: (text: string) => run(() => raw.encode(text)),
    decode: (tokens: readonly number[]) => run(() => raw.decode(tokens)),
    ...(measureTokens ? { measureTokens: (input: Parameters<typeof measureTokens>[0], signal?: AbortSignal) =>
      run(() => measureTokens.call(raw, input, signal), signal) } : {}),
    embed: (inputs: Parameters<EmbeddingCapability["embed"]>[0], signal?: AbortSignal) =>
      run(() => raw.embed(inputs, signal), signal),
    close(): Promise<void> {
      if (!closing) {
        const wait = active === 0 ? Promise.resolve() : new Promise<void>((resolve) => { drained = resolve; });
        closing = wait.then(() => raw.close());
      }
      return closing;
    },
  });
}
