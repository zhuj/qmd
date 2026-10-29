import { z } from "zod";

export interface EmbeddingExportProfile {
  readonly kind: "bert-encoder-v1";
  readonly hiddenSize: number;
  readonly maxInputTokens: number;
  readonly tokenizerClass: "BertTokenizer";
}
const modelSchema = z.object({ model_type: z.literal("bert"), hidden_size: z.number().int().positive(),
  max_position_embeddings: z.number().int().min(3), is_encoder_decoder: z.literal(false).optional(),
  is_decoder: z.literal(false).optional(), auto_map: z.never().optional(), trust_remote_code: z.literal(false).optional() });
const tokenizerSchema = z.object({ tokenizer_class: z.enum(["BertTokenizer", "BertTokenizerFast"]),
  model_max_length: z.number().positive().refine(Number.isInteger).optional(),
  padding_side: z.enum(["left", "right"]).optional(), truncation_side: z.enum(["left", "right"]).optional(),
  auto_map: z.never().optional(), trust_remote_code: z.literal(false).optional() });
const dataSchema = z.object({ model: z.object({ type: z.literal("WordPiece") }) });

/** This is the first known single-encoder contract, not an AutoModel registry
 * fallback or proof that arbitrary tokenizer data/ONNX computation is valid. */
export function parseEmbeddingExportProfile(
  model: unknown, tokenizerConfig: unknown, tokenizerData: unknown,
): EmbeddingExportProfile {
  const parsedModel = modelSchema.safeParse(model), parsedTokenizer = tokenizerSchema.safeParse(tokenizerConfig);
  if (!parsedModel.success || !parsedTokenizer.success || !dataSchema.safeParse(tokenizerData).success) {
    throw new Error("Unsupported embedding export profile: require a BERT encoder, explicit context/hidden size and a bundled WordPiece tokenizer without remote code");
  }
  const declared = parsedTokenizer.data.model_max_length;
  // Hugging Face uses enormous model_max_length sentinels for unspecified caps.
  // Only an explicit, safe tokenizer cap can tighten the required model limit.
  const tokenizerLimit = declared !== undefined && Number.isSafeInteger(declared) ? declared : Infinity;
  const maxInputTokens = Math.min(parsedModel.data.max_position_embeddings, tokenizerLimit);
  if (maxInputTokens < 3) throw new Error("Embedding export profile leaves no BERT body-token capacity");
  return Object.freeze({ kind: "bert-encoder-v1", hiddenSize: parsedModel.data.hidden_size,
    maxInputTokens, tokenizerClass: "BertTokenizer" });
}
