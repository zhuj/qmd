import { readFileSync } from "node:fs";
import { expect, test } from "vitest";
import { z } from "zod";

const manifest = z.object({
  dependencies: z.record(z.string(), z.string()),
  engines: z.object({ node: z.string() }),
  pnpm: z.object({ onlyBuiltDependencies: z.array(z.string()) }),
}).parse(JSON.parse(readFileSync(new URL("../package.json", import.meta.url), "utf8")));

test("ONNX direct dependency stays exact on the project's supported Node baseline", () => {
  expect(manifest.dependencies["@huggingface/transformers"]).toBe("4.3.1");
  expect(manifest.engines.node).toBe(">=22.0.0");
});

test.each(["@huggingface/transformers", "onnxruntime-node"])(
  "%s native lifecycle permission requires an explicit reviewed policy change", (dependency) => {
    expect(manifest.pnpm.onlyBuiltDependencies).not.toContain(dependency);
  },
);
