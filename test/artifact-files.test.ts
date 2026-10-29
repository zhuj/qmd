import { createHash } from "node:crypto";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { join, resolve } from "node:path";
import { expect, test } from "vitest";
import { createArtifactManifest } from "../src/embeddings/identity.js";
import { createArtifactStage, type ArtifactSource, type ArtifactStage } from "../src/embeddings/artifact-files.js";
import { deferred } from "./_helpers/deferred.js";

function source(...parts: readonly string[]): ArtifactSource {
  return {
    byteLength: parts.reduce((total, part) => total + Buffer.byteLength(part), 0),
    async *chunks(signal) { for (const part of parts) { signal?.throwIfAborted(); yield Buffer.from(part); } },
  };
}
async function withStage(body: (stage: ArtifactStage, root: string) => Promise<void>): Promise<void> {
  const root = await mkdtemp(resolve(".tmp", "onnx-files-"));
  try {
    const stage = await createArtifactStage(root);
    try { await body(stage, root); } finally { await stage.release(); }
  } finally { await rm(root, { recursive: true, force: true }); }
}

test("streams actual bytes into an owned stage with full hashes", async () => {
  await withStage(async (stage) => {
    const result = await stage.put({ path: "onnx/model.onnx", source: source("first", "λ", "last"), maxBytes: 100 });
    const expected = Buffer.from("firstλlast");
    expect(await readFile(join(stage.directory, "onnx/model.onnx"))).toEqual(expected);
    expect(result.entry).toEqual({ path: "onnx/model.onnx", bytes: expected.length,
      sha256: createHash("sha256").update(expected).digest("hex") });
    expect(result.sha1).toBe(createHash("sha1").update(expected).digest("hex"));
    const graph = stage.graph("onnx/model.onnx");
    expect(graph.byteLength).toBe(expected.length);
    expect(Buffer.from(await graph.read(5, 2))).toEqual(Buffer.from("λ"));
  });
});

test("seals only an exact manifest and stops new writes", async () => {
  await withStage(async (stage, root) => {
    const result = await stage.put({ path: "config.json", source: source("{}"), maxBytes: 100 });
    const manifest = createArtifactManifest({ kind: "local", path: root }, [result.entry]);
    await stage.seal(manifest);
    expect(JSON.parse(await readFile(join(stage.directory, ".qmd-manifest.json"), "utf8"))).toEqual(manifest);
    expect(Buffer.from(await stage.readMetadata("config.json")).toString()).toBe("{}");
    await expect(stage.put({ path: "extra.json", source: source("{}"), maxBytes: 100 })).rejects.toThrow(/seal|clos/i);
  });
});

test.each(["digest", "omitted", "renamed"])("rejects internally inconsistent %s manifest before sealing", async (change) => {
  await withStage(async (stage, root) => {
    const result = await stage.put({ path: "config.json", source: source("{}"), maxBytes: 100 });
    const valid = createArtifactManifest({ kind: "local", path: root }, [result.entry]);
    const files = change === "omitted" ? [] : [{ ...result.entry,
      path: change === "renamed" ? "other.json" : result.entry.path,
      sha256: change === "digest" ? "0".repeat(64) : result.entry.sha256 }];
    await expect(stage.seal({ ...valid, files })).rejects.toThrow();
    await stage.seal(valid);
    expect(JSON.parse(await readFile(join(stage.directory, ".qmd-manifest.json"), "utf8"))).toEqual(valid);
  });
});

test("rejects declared length mismatch and retains no partial file", async () => {
  await withStage(async (stage) => {
    await expect(stage.put({ path: "model.onnx", source: { ...source("abc"), byteLength: 4 }, maxBytes: 100 }))
      .rejects.toThrow(/length|size/i);
    expect(() => stage.graph("model.onnx")).toThrow(/missing|staged/i);
  });
});

test.each(["../outside", "/absolute", "a\\b", "NUL.bin", "data.", "nested./data"])(
  "rejects unsafe staged path %s", async (path) => {
    await withStage(async (stage) => {
      await expect(stage.put({ path, source: source("data"), maxBytes: 100 })).rejects.toThrow(/path|location/i);
    });
  },
);

test("failed intermediate directory creation cannot poison owned cleanup", async () => {
  await withStage(async (stage) => {
    await writeFile(join(stage.directory, "blocked"), "fixture");
    await expect(stage.put({ path: "blocked/more/model.onnx", source: source("abc"), maxBytes: 100 })).rejects.toThrow();
    await expect(stage.release()).resolves.toBeUndefined();
  });
});

test("rejects digest mismatch, oversized streams and case-folded duplicate paths", async () => {
  await withStage(async (stage) => {
    await expect(stage.put({ path: "bad.bin", source: { ...source("abc"), expectedSha256: "0".repeat(64) }, maxBytes: 100 }))
      .rejects.toThrow(/integrity/i);
    await expect(stage.put({ path: "large.bin", source: source("abc"), maxBytes: 2 })).rejects.toThrow(/size|length/i);
    await stage.put({ path: "valid.bin", source: source("abc"), maxBytes: 100 });
    await expect(stage.put({ path: "VALID.bin", source: source("other"), maxBytes: 100 })).rejects.toThrow(/duplicate/i);
  });
});

test("verifies Git blob digest without confusing it with raw SHA-1", async () => {
  await withStage(async (stage) => {
    const expectedGitSha1 = createHash("sha1").update("blob 3\0abc").digest("hex");
    const value = await stage.put({ path: "git.bin", source: { ...source("a", "bc"), expectedGitSha1 }, maxBytes: 100 });
    expect(value.sha1).not.toBe(expectedGitSha1);
    expect(value.entry.bytes).toBe(3);
  });
});

test("rejects a manifest that omits staged bytes", async () => {
  await withStage(async (stage, root) => {
    const first = await stage.put({ path: "first.bin", source: source("first"), maxBytes: 100 });
    await stage.put({ path: "second.bin", source: source("second"), maxBytes: 100 });
    await expect(stage.seal(createArtifactManifest({ kind: "local", path: root }, [first.entry])))
      .rejects.toThrow(/manifest/i);
  });
});

test("releasing one owned snapshot leaves another snapshot readable", async () => {
  await withStage(async (first, root) => {
    const second = await createArtifactStage(root);
    try {
      await second.put({ path: "config.json", source: source("{}"), maxBytes: 100 });
      await first.release();
      expect(Buffer.from(await second.readMetadata("config.json")).toString()).toBe("{}");
      await expect(first.put({ path: "closed.bin", source: source("data"), maxBytes: 100 })).rejects.toThrow(/clos/i);
    } finally { await second.release(); }
  });
});

test("release cancels an owned streaming source and waits for its cleanup", async () => {
  await withStage(async (stage) => {
    const started = deferred<void>();
    let sourceClosed = false;
    const streaming: ArtifactSource = {
      async *chunks(signal) {
        if (!signal) throw new Error("fixture requires stage-owned cancellation");
        try {
          yield Buffer.from("first");
          await new Promise<void>((_resolve, reject) => {
            signal.addEventListener("abort", () => reject(signal.reason), { once: true });
            started.resolve();
          });
        } finally { sourceClosed = true; }
      },
    };
    const writing = stage.put({ path: "pending.bin", source: streaming, maxBytes: 100 }).catch((error: unknown) => error);
    await started.promise;
    const closing = stage.release();
    expect(await writing).toEqual(new Error("Artifact stage closing"));
    await closing;
    expect(sourceClosed).toBe(true);
  });
});
