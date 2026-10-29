import { createEmbeddingStoreFixture, fixtureEmbeddingConfig } from "./embedding-store.js";
import { makeSpaceDescriptor } from "./embedding-config.js";
import { createArtifactManifest, embeddingSpaceId } from "../../src/embeddings/identity.js";
import { measureEmbeddingInput } from "../../src/embeddings/recipe.js";
import type { EmbeddingCapability, EmbeddingInput, ResolvedOnnxConfig } from "../../src/embeddings/types.js";

export function controlledIndexingFixture(beforeEmbed?: (inputs: readonly EmbeddingInput[]) => Promise<void>, device: ResolvedOnnxConfig["device"] = "cpu") {
  const base = fixtureEmbeddingConfig();
  const config: ResolvedOnnxConfig = { ...base, device, fallback: "cpu", recipe: { ...base.recipe, maxInputTokens: 8, titleBudgetTokens: 0, overlapTokens: 0 }, maxBatchTokens: 16, batchSize: 2 };
  const original = makeSpaceDescriptor();
  const descriptor = { ...original, artifactManifest: createArtifactManifest(config.source, original.artifactManifest.files), recipe: config.recipe,
    chunking: { ...original.chunking, maxInputTokens: 8, titleBudgetTokens: 0, overlapTokens: 0 } };
  const observed: EmbeddingInput[][] = [];
  const tokenizer = {
    encode: async (text: string, special: boolean) => special ? [0, ...Array.from(text, c => c.codePointAt(0) ?? 0), 0] : Array.from(text, c => c.codePointAt(0) ?? 0),
    decode: async (tokens: readonly number[]) => String.fromCodePoint(...tokens),
  };
  const capability: EmbeddingCapability = {
    config, modelLabel: "controlled-onnx", provenance: device === "cpu" ? undefined : {
      requestedDevice: device, initializedMode: "cpu", graphNodePlacement: "unknown", cpuRetryReason: "provider-initialization-failed",
    },
    prepare: async () => ({ kind: "onnx-v1", id: embeddingSpaceId(descriptor), descriptor }),
    encode: text => tokenizer.encode(text, true), decode: tokens => tokenizer.decode(tokens),
    measureTokens: input => measureEmbeddingInput(input, config.recipe, tokenizer),
    embed: async inputs => {
      observed.push([...inputs]); await beforeEmbed?.(inputs);
      return inputs.map(input => new Float32Array([input.text.length + 1, 1]));
    },
    close: async () => {},
  };
  const fixture = createEmbeddingStoreFixture(config, capability);
  return { ...fixture, calls: observed, descriptor };
}
