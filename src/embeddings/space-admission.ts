import type { EmbeddingIdentity } from "./types.js";
import { embeddingSpaceId } from "./identity.js";

export type StoredEmbeddingSpace =
  | Readonly<{ kind: "empty" }>
  | Extract<EmbeddingIdentity, { kind: "legacy-gguf" }>
  | Readonly<{ kind: "onnx-v1"; identity: Extract<EmbeddingIdentity, { kind: "onnx-v1" }> }>
  | Readonly<{ kind: "ambiguous"; reason: string }>;
export interface EmbeddingAdmission {
  readonly action: "read" | "compatible-write" | "initialize" | "legacy-full-force";
  readonly identity: EmbeddingIdentity;
  readonly expected: string;
}
export interface EmbeddingOperation { readonly kind: "query" | "embed"; readonly force?: boolean; readonly collection?: string }
function assertNever(value: never): never { throw new Error(`Unknown embedding state: ${String(value)}`); }
function assertIdentity(identity: EmbeddingIdentity): void {
  switch (identity.kind) {
    case "onnx-v1":
      if (embeddingSpaceId(identity.descriptor) !== identity.id) throw new Error("Embedding identity hash does not match its descriptor");
      return;
    case "legacy-gguf":
      if (!identity.model || !/^[a-f0-9]{6}$/.test(identity.fingerprint) || !Number.isSafeInteger(identity.dimensions) || identity.dimensions < 1) {
        throw new Error("Invalid legacy embedding identity");
      }
      return;
    default: return assertNever(identity);
  }
}
export function embeddingStateKey(state: StoredEmbeddingSpace): string {
  switch (state.kind) {
    case "empty": return "empty";
    case "ambiguous": return JSON.stringify([state.kind, state.reason]);
    case "legacy-gguf": assertIdentity(state); return JSON.stringify([state.kind, state.model, state.fingerprint, state.dimensions]);
    case "onnx-v1": assertIdentity(state.identity); return state.identity.id;
    default: return assertNever(state);
  }
}
export function planEmbeddingAdmission(current: StoredEmbeddingSpace, producer: EmbeddingIdentity, operation: EmbeddingOperation): EmbeddingAdmission {
  assertIdentity(producer);
  const expected = embeddingStateKey(current);
  const compatible = operation.kind === "query" ? "read" : "compatible-write";
  switch (current.kind) {
    case "empty": return Object.freeze({ action: operation.kind === "query" ? "read" : "initialize", identity: producer, expected });
    case "ambiguous": throw new Error("Ambiguous embedding identity; refusing vector operations");
    case "onnx-v1":
      if (producer.kind !== "onnx-v1" || producer.id !== current.identity.id) {
        throw new Error("Embedding space changed; use a separate database");
      }
      return Object.freeze({ action: compatible, identity: producer, expected });
    case "legacy-gguf": {
      if (producer.kind !== "legacy-gguf") throw new Error("Embedding backend/space changed; use a separate database");
      const matches = producer.model === current.model && producer.fingerprint === current.fingerprint && producer.dimensions === current.dimensions;
      if (matches) return Object.freeze({ action: compatible, identity: producer, expected });
      if (operation.kind === "embed" && operation.force === true && operation.collection === undefined) {
        return Object.freeze({ action: "legacy-full-force", identity: producer, expected });
      }
      throw new Error("Legacy embedding identity changed; matching producer or explicit unscoped full force is required");
    }
    default: return assertNever(current);
  }
}
