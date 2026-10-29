import type { OnnxRuntimeSession } from "./onnx-runtime-types.js";
import type { EmbeddingModelOutput } from "./recipe.js";
import { z } from "zod";
import { nativeFunction, nativeMethod, nativeObject, nativeProperty, NativeTensorScope, readInt64Tensor, readFloatTensor } from "./onnx-native-values.js";

export interface TransformersSessionObjects {
  readonly tokenizer: unknown;
  readonly model: unknown;
  int64Tensor(data: BigInt64Array, dims: readonly number[]): unknown;
}

export function createNativeEmbeddingSession(objects: TransformersSessionObjects): OnnxRuntimeSession {
  try { return admitNativeEmbeddingSession(objects); }
  catch { throw new Error("Unsupported ONNX session contract"); }
}

function admitNativeEmbeddingSession(objects: TransformersSessionObjects): OnnxRuntimeSession {
  const sessions = nativeObject(nativeProperty(objects.model, "sessions"));
  if (Object.keys(sessions).length !== 1 || !Object.hasOwn(sessions, "model")) throw new Error("Unsupported ONNX model session contract");
  const session = nativeProperty(sessions, "model"), run = nativeMethod(session, "run");
  const parsedNames = z.array(z.enum(["input_ids", "attention_mask", "token_type_ids"])).min(2).max(3).safeParse(nativeProperty(session, "inputNames"));
  if (!parsedNames.success) throw new Error("Unsupported ONNX input contract");
  const names = parsedNames.data;
  if (!names.includes("input_ids") || !names.includes("attention_mask") || new Set(names).size !== names.length) throw new Error("Unsupported ONNX input contract");
  const tokenize = nativeFunction(objects.tokenizer), encode = nativeMethod(tokenize, "encode"), decode = nativeMethod(tokenize, "decode");
  const disposeModel = nativeMethod(objects.model, "dispose");
  let active = 0, drained: (() => void) | undefined, closing: Promise<void> | undefined;
  const begin = () => {
    if (closing) throw new Error("ONNX session closing");
    active++;
    let finished = false;
    return () => { if (!finished) { finished = true; if (--active === 0) drained?.(); } };
  };
  const tokenIds = z.array(z.number().int().nonnegative().max(2147483647));
  return Object.freeze({
    tokenizer: Object.freeze({
      encode: async (text: string, addSpecialTokens: boolean) => {
        const finish = begin();
        try { return Object.freeze(tokenIds.parse(await encode(text, { add_special_tokens: addSpecialTokens }))); }
        catch { throw new Error("ONNX tokenization failed"); } finally { finish(); }
      },
      decode: async (tokens: readonly number[]) => {
        const finish = begin();
        try {
          const result = await decode(tokenIds.parse(tokens), { skip_special_tokens: false, clean_up_tokenization_spaces: false });
          if (typeof result !== "string") throw new Error("Invalid decoded text");
          return result;
        } catch { throw new Error("ONNX tokenization failed"); } finally { finish(); }
      },
    }),
    run: async (texts: readonly string[]) => {
      const finish = begin(), scope = new NativeTensorScope();
      let delivered = false;
      try {
        const batch: unknown = await tokenize.call(objects.tokenizer, [...texts], { padding: true, truncation: false, return_tensor: true });
        scope.inputs(batch);
        const ids = readInt64Tensor(scope.transformer(nativeProperty(batch, "input_ids")));
        const attention = readInt64Tensor(scope.transformer(nativeProperty(batch, "attention_mask")));
        const sequence = ids.dims[1];
        if (sequence === undefined || ids.dims[0] !== texts.length || ids.dims.join() !== attention.dims.join()
          || ids.data.some(value => value < 0n || value > 2147483647n)) throw new Error("Invalid tokenizer batch");
        const mask: number[][] = [];
        for (let row = 0; row < texts.length; row++) {
          const values = Array.from(attention.data.subarray(row * sequence, (row + 1) * sequence), value => {
            if (value !== 0n && value !== 1n) throw new Error("Invalid attention mask");
            return value === 1n ? 1 : 0;
          });
          if (!values.includes(1)) throw new Error("Invalid attention mask");
          mask.push(values);
        }
        const feeds = Object.fromEntries(names.map(name => {
          const value = nativeProperty(batch, name);
          const native = name === "token_type_ids" && value === undefined
            ? scope.transformer(objects.int64Tensor(new BigInt64Array(ids.data.length), ids.dims)) : scope.transformer(value);
          const tensor = readInt64Tensor(native);
          if (tensor.dims.join() !== ids.dims.join()) throw new Error("Invalid input dimensions");
          return [name, native];
        }));
        const results: unknown = await run(feeds);
        scope.outputs(results);
        const output: EmbeddingModelOutput = {
          ...(Object.hasOwn(nativeObject(results), "last_hidden_state") ? { last_hidden_state: readFloatTensor(nativeProperty(results, "last_hidden_state")) } : {}),
          ...(Object.hasOwn(nativeObject(results), "sentence_embedding") ? { sentence_embedding: readFloatTensor(nativeProperty(results, "sentence_embedding")) } : {}),
        };
        if (!output.last_hidden_state && !output.sentence_embedding) throw new Error("Missing embedding output");
        let releasing: Promise<void> | undefined;
        const release = () => releasing ??= scope.release().finally(finish);
        delivered = true;
        return { output, mask, release };
      } catch { throw new Error("ONNX inference failed"); }
      finally {
        if (!delivered) {
          try { await scope.release(); } catch { /* Preserve the sanitized inference failure over secondary cleanup. */ }
          finally { finish(); }
        }
      }
    },
    close: () => closing ??= Promise.resolve().then(async () => {
      if (active) await new Promise<void>(resolve => { drained = resolve; });
      try { await disposeModel(); } catch { throw new Error("ONNX session cleanup failed"); }
    }),
  });
}
