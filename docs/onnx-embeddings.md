---
title: ONNX Embeddings — Implemented Behavior (Phase 1 T5–T16)
version: v1
status: reconciled
approved: 2026-10-23
baseline: 86d2365
---

# ONNX Embeddings — Actual Implemented Behavior

> Reconciled with T5–T16; documentation only — no new runtime claims.

## 1. What is implemented (not promised)

- Configurable ONNX descriptor (`backend: "onnx"`) with `source` (`huggingface` with immutable `revision`, or `local` with absolute/relative-authorized path), `recipe` (`version`, `pooling`, `normalization`, prefixes, `maxInputTokens`/`titleBudgetTokens`/`overlapTokens`), `device` (`cpu`/`cuda`/`dml`/`coreml`), `dtype` (`fp32`/`q8`), `dimensions`, `fallback`, `allowDownload`, `batchSize`/`maxBatchTokens`/`intraOpNumThreads`.
- Per-store immutable descriptor/hash in `store_config` (`embedding_space_v1`); identity covers backend + artifact manifest + recipe + chunking + dtype + dims; same-dimension different descriptor = incompatible (rejected before clear/write); one active space per DB.
- Legacy GGUF full-index `qmd embed -f` preserved unscoped (equal or changed dimensions, validated preflight, `clearAllEmbeddings()`); ONNX does not adopt legacy vectors by matching dims alone; lexical stays usable on failure/preflight/cancel (fail-closed).
- CPU-default search: original-query BM25 + vector deterministic fusion (RRF, dedup, tie-break); `expand`/`rerank` disabled when `backend === "onnx" && device === "cpu"`; explicit stage-enable conflict errors; CPU retry never rewrites configured-stage policy; accelerated ONNX keeps independent stage settings.
- Cancellation stops queued/new work; cancelled batch cannot commit; inflight settles before session release; store close waits for leased inference.
- Diagnostic (`qmd doctor`) uses fail-closed producer-relative health: `EmbeddingCompatibilityUnverifiedError` when no verified producer proof (`compatible = false` → `unverified` semantics); descriptor/member/reason redacted; reasons not exposed to unverified callers.
- Packaging/installer: exact pins `@huggingface/transformers@4.3.1` with lockfile-resolved `onnxruntime-node@1.30.0`; the upstream `ONNXRUNTIME_NODE_INSTALL=skip` install-script opt-out was the documented setup policy (device config `cpu` does not control or prevent package-install downloads); CUDA requires separately provisioned provider binaries + compatible hardware/driver; Bun/native-API/platform compatibility unverified (recorded honestly, not claimed).

## 2. Recipe / descriptor fields (actual, per design spec)

- Input config (z-validated, `src/embeddings/config.ts`): `backend: "onnx"` required; `source`, `recipe` required; `device` (`cpu`/`cuda`/`dml`/`coreml`, default `cpu`), `fallback` (`error`/`cpu`, default `error`), `dtype` (`fp32`/`q8`, default `fp32`), `dimensions` (optional — omitted dimensions are measured from the native output), `allowDownload` (default `false`), `intraOpNumThreads` (default 1, capped at host parallelism), `batchSize` (default 1, max 32), `maxBatchTokens` (default 4096, max 65536).
- `recipe`: `{ version: "qmd-embedding-v1", pooling: "mean" | "cls" | "last-token" | "sentence-output", normalization: "l2" | "none", queryPrefix: string, documentPrefix: string, maxInputTokens, titleBudgetTokens, overlapTokens }`. Conditional requirements: `"sentence-output"` requires `sentenceOutput: "sentence_embedding"`; `"cls"` requires `clsTokenIndex` (`src/embeddings/types.ts`).
- Candidate explanation (not default): mean / `l2`; 256 total, 32 title, 24 overlap; no prefixes (`queryPrefix: ""`, `documentPrefix: ""`). Must be paired with a verified immutable HF `revision` chosen at T18 runtime qualification; not presented as usable without it.
- No arbitrary dimension reduction: the persisted space cannot be used unless `nativeDimensions === dimensions` (enforced at `src/embeddings/identity.ts:84`; a requested different output size is rejected).
- Artifact manifest is built and validated by preparation from the selected graph closure (`config.json`, `tokenizer_config.json`, `tokenizer.json`, the dtype-selected graph, and its external data; `src/embeddings/artifacts.ts`): users do not supply the manifest file — a local directory must contain the ONNX export artifacts themselves (including referenced external data); missing/partial closures fail preflight. The graph file is selected by `dtype`: `onnx/model.onnx` for `fp32`, `onnx/model_quantized.onnx` for `q8`.

## 3. Space / error semantics

- `embeddingSpaceDescriptor` persisted as `embedding_space_v1` in `store_config`; admission (`planEmbeddingAdmission`) compares the full descriptor per store operation; failure = existing DB preserved, lexical usable (`search`/`searchLex` via DB); force requires validated preflight first.
- Same dimensions with a different descriptor (recipe/source/dtype change) is incompatible and rejected — see `test/onnx-integration.test.ts`.
- `compatible = false` → `unverified` means "not established", not proven incompatible; diagnostics redact paths/raw text and give separate-database guidance on ambiguity.

## 4. CPU search policy (actual)

- Original query sent to BM25 and vector; fusion deterministic; no HyDE/expansion when CPU; no neural rerank when CPU; caller-supplied `vec`/`lex`/`hyde` routed exclusively (`lex`→BM25, `vec`/`hyde`→vector); conflicting stage requests → error (do not load).
- CPU retry on provider-init failure retries once with visible provenance (`requestedDevice`, `initializedMode: "cpu" | "cuda" | "dml" | "coreml", cpuRetryReason?`); never rewrites interface-stage policy.

## 5. Usable local configuration example (explanation + shape, not a qualified model)

```yaml
models:
  embed:
    backend: onnx
    source:
      kind: local
      path: ./models/authorized-local-export   # relative paths resolve against the config file directory
    recipe:
      version: qmd-embedding-v1
      pooling: mean            # candidate recipe (explanation only; pairs with a verified T18 revision)
      normalization: l2
      queryPrefix: ""
      documentPrefix: ""
      maxInputTokens: 256
      titleBudgetTokens: 32
      overlapTokens: 24
    device: cpu
    dtype: fp32                # selects onnx/model.onnx (q8 selects onnx/model_quantized.onnx)
    # dimensions omitted here are measured from the native output; no arbitrary reduction is supported
    allowDownload: false       # default false: missing/cached artifacts fail preflight offline
```

- Inline SDK configuration needs ABSOLUTE local paths (relative `source.path` without a config-file context errors, `src/embeddings/config.ts:69`); YAML paths may be config-relative.
- Offline is the default acquisition policy: on cache miss with `allowDownload: false` preparation fails fail-closed (`ONNX artifacts are not cached; offline mode forbids acquisition`, `src/embeddings/artifacts.ts`); only explicit `allowDownload: true` opts into acquisition.
- The candidate mean/l2 256/32/24 recipe is EXPLANATION ONLY. A usable published HF example requires a verified immutable revision chosen in the deferred T18 runtime-qualification task; local authorized directory configuration is shown instead. No all-zero/fake SHA is presented as a model reference. CLS (`clsTokenIndex`) / attention-mask-aware `last-token` / sentence-output (`sentenceOutput: "sentence_embedding"`) are bounded explicit recipes, not universal semantics.

## 6. Installer / packaging (honest)

- Exact pins: `@huggingface/transformers@4.3.1`; the lockfile resolves `onnxruntime-node@1.30.0` (has an install script) and `onnxruntime-web@1.31.0-dev.20260914-8d85527a0` (`bun.lock`).
- `ONNXRUNTIME_NODE_INSTALL` is the UPSTREAM onnxruntime-node install-script mechanism; the documented setup `ONNXRUNTIME_NODE_INSTALL=skip BUN_INSTALL_CACHE_DIR=... bun add --exact @huggingface/transformers@4.3.1 --ignore-scripts` skips that install script's acquisition work. Device config `cpu` controls inference device selection, NOT package-install downloads — never imply CPU device config prevents them.
- Explicit CUDA use requires separately provisioned provider binaries, compatible hardware and driver; the CPU-oriented opt-out above does not provide them.
- Runtime is recorded honestly; no independent Bun/platform/native-API qualification claim is made (T18 scope).

## 7. What is DELIBERATELY NOT claimed

- No hardware-support claim (T18 gates CUDA/DML/CoreML).
- No dtype/quantization claim beyond `fp32`/`q8` selection.
- No performance/relevance improvement.
- No automatic model selection; explicit descriptor required.
- No cross-backend adoption of legacy vectors.
