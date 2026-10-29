import { expect, test } from "vitest";
import { parseEmbeddingExportProfile } from "../src/embeddings/artifact-profile.js";

const model = { model_type: "bert", hidden_size: 384, max_position_embeddings: 512 };
const tokenizerConfig = { tokenizer_class: "BertTokenizer", model_max_length: 1e30 };
const tokenizerData = { model: { type: "WordPiece", vocab: { "[UNK]": 0 }, unk_token: "[UNK]" } };
test("known BERT metadata supplies explicit context and hidden size without tokenizer sentinels", () => {
  const profile = parseEmbeddingExportProfile(model, tokenizerConfig, tokenizerData);
  expect(profile).toEqual({ kind: "bert-encoder-v1", hiddenSize: 384, maxInputTokens: 512, tokenizerClass: "BertTokenizer" });
  expect(Object.isFrozen(profile)).toBe(true);
  expect(model).toEqual({ model_type: "bert", hidden_size: 384, max_position_embeddings: 512 });
});
test("the known Fast alias has identical export semantics", () => {
  expect(parseEmbeddingExportProfile(model, { ...tokenizerConfig, tokenizer_class: "BertTokenizerFast", padding_side: "left" }, tokenizerData))
    .toEqual(parseEmbeddingExportProfile(model, tokenizerConfig, tokenizerData));
});
test("a real tokenizer cap can only tighten a known model context", () => {
  expect(parseEmbeddingExportProfile(model, { ...tokenizerConfig, model_max_length: 64 }, tokenizerData).maxInputTokens).toBe(64);
  expect(parseEmbeddingExportProfile(model, { ...tokenizerConfig, model_max_length: 1024 }, tokenizerData).maxInputTokens).toBe(512);
  expect(parseEmbeddingExportProfile(model, { tokenizer_class: "BertTokenizer" }, tokenizerData).maxInputTokens).toBe(512);
});
test.each([
  { model_type: "unknown" }, { model_type: "gpt2" }, { model_type: undefined },
  { hidden_size: 0 }, { hidden_size: -1 }, { hidden_size: 1.5 }, { hidden_size: Infinity },
  { max_position_embeddings: undefined }, { max_position_embeddings: 2 }, { max_position_embeddings: 1.5 },
  { is_encoder_decoder: true }, { is_decoder: true }, { auto_map: { AutoModel: "custom.Model" } }, { trust_remote_code: true },
])("rejects unknown or incompatible model contracts: %j", change => {
  expect(() => parseEmbeddingExportProfile({ ...model, ...change }, tokenizerConfig, tokenizerData)).toThrow(/profile|export/i);
});
test.each([
  { tokenizer_class: undefined }, { tokenizer_class: "UnknownTokenizer" }, { tokenizer_class: "RobertaTokenizer" },
  { padding_side: "other" }, { truncation_side: "other" },
  { model_max_length: 0 }, { model_max_length: -1 }, { model_max_length: 1.5 }, { model_max_length: Infinity }, { model_max_length: 2 },
  { auto_map: { AutoTokenizer: "custom.Tokenizer" } }, { trust_remote_code: true },
])("rejects implicit tokenizer fallback or remote code: %j", change => {
  expect(() => parseEmbeddingExportProfile(model, { ...tokenizerConfig, ...change }, tokenizerData)).toThrow(/profile|export/i);
});
test.each([{ data: {} }, { data: { model: { type: "BPE" } } }, { data: null }])("requires the known tokenizer data contract", ({ data }) => {
  expect(() => parseEmbeddingExportProfile(model, tokenizerConfig, data)).toThrow(/profile|export/i);
});
