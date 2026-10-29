import type { Database } from "../db.js";
import type { RuntimeProvenance } from "./types.js";
import type { EmbeddingAdmission, StoredEmbeddingSpace } from "./space-admission.js";
import { embeddingStateKey, planEmbeddingAdmission } from "./space-admission.js";
import { embeddingSpaceId, parseEmbeddingSpaceDescriptor } from "./identity.js";
import { vecLayout, VEC_ROWS_TABLE } from "../vec-layout.js";
import { z } from "zod";
export { planEmbeddingAdmission } from "./space-admission.js";
export type { EmbeddingAdmission, StoredEmbeddingSpace } from "./space-admission.js";

const SPACE_KEY = "embedding_space_v1", RUNTIME_KEY = "embedding_runtime_v1", JSON_LIMIT = 2 * 1024 * 1024;
const storedSchema = z.strictObject({ version: z.literal(1), id: z.string(), descriptor: z.unknown() });
const labelsSchema = z.array(z.object({ model: z.string(), embed_fingerprint: z.string() }));
const runtimeSchema = z.strictObject({ requestedDevice: z.enum(["cpu", "cuda", "dml", "coreml"]),
  initializedMode: z.enum(["cpu", "cuda", "dml", "coreml"]), graphNodePlacement: z.literal("unknown"),
  cpuRetryReason: z.literal("provider-initialization-failed").optional() });
function ambiguous(reason: string): StoredEmbeddingSpace { return Object.freeze({ kind: "ambiguous", reason }); }
function count(db: Database, sql: string): number {
  return z.object({ n: z.number().int().nonnegative() }).parse(db.prepare(sql).get()).n;
}
function recognizedLegacySelector(model: string): boolean {
  if (!model || /[\x00-\x1f]/.test(model)) return false;
  if (model.startsWith("hf:")) return /^hf:[^\s/]+\/[^\s/]+\/.+\.gguf$/i.test(model);
  if (/^https?:\/\//i.test(model)) {
    try { return /\.gguf$/i.test(new URL(model).pathname); } catch { return false; }
  }
  return !/^[a-z]+:/i.test(model.replace(/^[a-z]:[\\/]/i, "")) && /\.gguf$/i.test(model);
}
function parseJson(value: unknown): unknown {
  if (typeof value !== "string" || Buffer.byteLength(value) > JSON_LIMIT) throw new Error("Invalid embedding metadata");
  const parsed: unknown = JSON.parse(value); return parsed;
}
export function readEmbeddingSpace(db: Database): StoredEmbeddingSpace {
  try {
    const entries = db.prepare("SELECT key,value FROM store_config WHERE key IN (?,?)").all<{ key: string; value: unknown }>(SPACE_KEY, RUNTIME_KEY);
    const stored = entries.find(row => row.key === SPACE_KEY), runtime = entries.find(row => row.key === RUNTIME_KEY);
    const labels = labelsSchema.parse(db.prepare("SELECT model,embed_fingerprint FROM content_vectors GROUP BY model,embed_fingerprint LIMIT 2").all());
    const layout = vecLayout(db);
    const mappings = count(db, `SELECT COUNT(*) AS n FROM ${VEC_ROWS_TABLE}`);
    const physical = layout.kind === "none" ? 0 : count(db, `SELECT COUNT(*) AS n FROM ${layout.table}`);
    if (layout.kind !== "none" && (!Number.isSafeInteger(layout.dimensions) || layout.dimensions === null || layout.dimensions < 1)) return ambiguous("Unknown vector dimensions");
    if (layout.kind === "none" && (labels.length > 0 || mappings > 0)) return ambiguous("Vector metadata has no readable index");
    if (labels.length === 0 && (physical > 0 || mappings > 0)) return ambiguous("Orphan physical vectors");
    if (layout.kind === "partitioned" && mappings !== physical) return ambiguous("Vector mapping/index row counts differ");
    if (layout.kind === "partitioned" && db.prepare(`SELECT 1 FROM ${VEC_ROWS_TABLE} r
      WHERE NOT EXISTS (SELECT 1 FROM ${layout.table} v WHERE v.rowid=r.id)
      OR NOT EXISTS (SELECT 1 FROM content_vectors c WHERE c.hash=r.hash AND c.seq=r.seq) LIMIT 1`).get()) {
      return ambiguous("Vector row mapping does not resolve to its index and content identity");
    }
    if (stored) {
      const parsed = storedSchema.parse(parseJson(stored.value)), descriptor = parseEmbeddingSpaceDescriptor(parsed.descriptor);
      if (embeddingSpaceId(descriptor) !== parsed.id) return ambiguous("Stored embedding identity hash mismatch");
      if (layout.kind !== "none" && layout.dimensions !== descriptor.dimensions) return ambiguous("Stored embedding dimension mismatch");
      if (labels.some(row => row.embed_fingerprint !== parsed.id)) return ambiguous("Vector rows do not match active identity");
      if (runtime) runtimeSchema.parse(parseJson(runtime.value));
      return Object.freeze({ kind: "onnx-v1", identity: Object.freeze({ kind: "onnx-v1", id: parsed.id, descriptor }) });
    }
    if (runtime) return ambiguous("Runtime provenance has no semantic descriptor");
    if (labels.length === 0) return Object.freeze({ kind: "empty" });
    const label = labels[0];
    if (labels.length !== 1 || !label || !/^[a-f0-9]{6}$/.test(label.embed_fingerprint)
      || !recognizedLegacySelector(label.model) || layout.kind === "none" || layout.dimensions === null) {
      return ambiguous("Unknown, mixed or unadopted legacy identity");
    }
    return Object.freeze({ kind: "legacy-gguf", model: label.model, fingerprint: label.embed_fingerprint, dimensions: layout.dimensions });
  } catch { return ambiguous("Embedding metadata or vector layout cannot be verified"); }
}
export function commitEmbeddingSpace(db: Database, admission: EmbeddingAdmission, provenance?: RuntimeProvenance): void {
  if (!db.inTransaction) throw new Error("Embedding metadata requires a caller-owned immediate transaction");
  if (admission.action === "read") throw new Error("Read admission cannot commit embedding metadata");
  const current = readEmbeddingSpace(db);
  if (embeddingStateKey(current) !== admission.expected) throw new Error("Embedding space changed since admission planning");
  const checked = planEmbeddingAdmission(current, admission.identity, { kind: "embed", force: admission.action === "legacy-full-force" });
  if (checked.action !== admission.action) throw new Error("Embedding admission action no longer matches current state");
  const identity = admission.identity;
  if (identity.kind === "legacy-gguf") return;
  const descriptor = parseEmbeddingSpaceDescriptor(identity.descriptor), layout = vecLayout(db);
  if (layout.kind !== "none" && layout.dimensions !== descriptor.dimensions) throw new Error("Existing vector dimensions require a separate database");
  const runtime = provenance === undefined ? undefined : JSON.stringify(runtimeSchema.parse(provenance));
  const semantic = JSON.stringify({ version: 1, id: identity.id, descriptor });
  if (Buffer.byteLength(semantic) > JSON_LIMIT) throw new Error("Embedding descriptor exceeds metadata size limit");
  if (admission.action === "initialize") db.prepare("INSERT INTO store_config(key,value) VALUES (?,?)").run(SPACE_KEY, semantic);
  if (runtime !== undefined) db.prepare("INSERT INTO store_config(key,value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value").run(RUNTIME_KEY, runtime);
}
