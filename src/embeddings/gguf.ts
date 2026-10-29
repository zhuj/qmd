import type { Token as LlamaToken } from "node-llama-cpp";
import { withLLMSessionForLlm, type EmbeddingResult, type LlamaCpp } from "../llm.js";
import type { EmbeddingCapability, EmbeddingIdentity, EmbeddingInput } from "./types.js";

export interface LegacyEmbeddingFormatting {
  query(text: string, model: string): string;
  document(text: string, title: string, model: string): string;
  fingerprint(model: string): string;
}

type LegacyEmbeddingIdentity = Extract<EmbeddingIdentity, { kind: "legacy-gguf" }>;

export function createGgufEmbeddingCapability(
  llm: LlamaCpp, model: string, formatting: LegacyEmbeddingFormatting,
): EmbeddingCapability {
  function assertProducer(): void {
    if (model !== llm.embedModelName) {
      throw new Error("Embedding model must match the configured GGUF producer; configure the store's model instead");
    }
  }
  assertProducer();
  const formats = Object.freeze({ ...formatting });
  let preparation: Promise<LegacyEmbeddingIdentity> | undefined;

  function vector(result: EmbeddingResult | null, dimensions?: number): Float32Array {
    if (!result || result.model !== model) throw new Error("Missing embedding or mismatched GGUF producer");
    const values = new Float32Array(result.embedding);
    if (values.length === 0 || values.some((value) => !Number.isFinite(value)) || !values.some((value) => value !== 0)) {
      throw new Error("GGUF embedding must be a finite nonzero Float32 vector");
    }
    if (dimensions !== undefined && values.length !== dimensions) throw new Error("GGUF embedding dimension changed");
    return values;
  }

  function format(input: EmbeddingInput): string {
    switch (input.kind) {
      case "query": return formats.query(input.text, model);
      case "document": return formats.document(input.text, input.title, model);
      default: return assertNever(input);
    }
  }

  async function prepare(signal?: AbortSignal): Promise<LegacyEmbeddingIdentity> {
    assertProducer();
    signal?.throwIfAborted();
    // Shared initialization belongs to this capability, not its first caller's abort signal.
    preparation ??= withLLMSessionForLlm(llm, async (session) => {
      const values = vector(await session.embed(formats.query("qmd embedding dimension probe", model)));
      return Object.freeze({ kind: "legacy-gguf", model, fingerprint: formats.fingerprint(model), dimensions: values.length });
    }, { name: "embedding-preflight", maxDuration: 0 }).catch((error: unknown) => {
      preparation = undefined;
      throw error;
    });
    const identity = await preparation;
    signal?.throwIfAborted();
    return identity;
  }

  return Object.freeze({
    config: Object.freeze({ backend: "gguf", model }), modelLabel: model, provenance: undefined,
    prepare,
    async encode(text: string): Promise<readonly number[]> {
      assertProducer();
      return withLLMSessionForLlm(llm, () => llm.tokenize(text), { name: "embedding-tokenize", maxDuration: 0 });
    },
    async decode(tokens: readonly number[]): Promise<string> {
      assertProducer();
      const nativeTokens = tokens.map((token) => {
        if (!Number.isInteger(token) || token < 0 || token >= 2 ** 31) throw new Error("Invalid GGUF token id");
        // The native signed token ID has an opaque compile-time brand, no public constructor.
        return token as LlamaToken;
      });
      return withLLMSessionForLlm(llm, () => llm.detokenize(nativeTokens), { name: "embedding-detokenize", maxDuration: 0 });
    },
    async embed(inputs: readonly EmbeddingInput[], signal?: AbortSignal): Promise<readonly Float32Array[]> {
      assertProducer();
      signal?.throwIfAborted();
      if (inputs.length === 0) return [];
      const texts = inputs.map(format);
      const identity = await prepare(signal);
      return withLLMSessionForLlm(llm, async (session) => {
        signal?.throwIfAborted();
        const results = await session.embedBatch(texts);
        signal?.throwIfAborted();
        if (results.length !== texts.length) throw new Error("GGUF embedding batch result count changed");
        return results.map((result) => vector(result, identity.dimensions));
      }, { name: "embedding-batch", maxDuration: 0 });
    },
    // The store, not this raw bridge, owns the borrowed LlamaCpp. T7 wraps this in a lease exactly once.
    close: async () => {},
  });
}

function assertNever(value: never): never {
  throw new Error(`Unexpected embedding input: ${String(value)}`);
}
