import { expect, test } from "vitest";
import { spawnSync } from "node:child_process";

test("SDK import and lexical store do not evaluate the AI runtime module", () => {
  // A separate Bun VM prevents the import sentinel from changing this test process's module registry.
  const script = `
    import { mock } from "bun:test";
    let imports = 0;
    mock.module("@huggingface/transformers", () => { imports++; throw new Error("fixture-runtime-import"); });
    const { createStore } = await import("./src/index.ts");
    const { makeOnnxConfig } = await import("./test/_helpers/embedding-config.ts");
    const store = await createStore({ dbPath: ":memory:", config: { collections: {}, models: {
      embed: makeOnnxConfig({ source: { kind: "local", path: "/fixture/missing-model" } }),
    } } });
    try { await store.searchLex("alpha"); } finally { await store.close(); }
    if (imports !== 0) throw new Error("Runtime evaluated during lexical construction");
    let intercepted = false;
    try { await import("@huggingface/transformers"); }
    catch (error) { intercepted = error instanceof Error && error.message === "fixture-runtime-import"; }
    if (!intercepted || imports !== 1) throw new Error("Import sentinel was not active");
    console.log("lexical-runtime-imports=0;sentinel=verified");
  `;
  const result = spawnSync("bun", ["--eval", script], { cwd: process.cwd(),
    env: { ...process.env, CI: "true" }, encoding: "utf8", timeout: 20_000, maxBuffer: 65_536 });
  expect(result.error).toBeUndefined();
  expect(result.status, result.stderr).toBe(0);
  expect(result.stdout.trim()).toBe("lexical-runtime-imports=0;sentinel=verified");
});
