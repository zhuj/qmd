import { describe, expect, test } from "vitest";
import { buildArgs, selectorsPresent, childExitCode } from "../scripts/test-bun.mjs";

describe("test-bun wrapper contract", () => {
  test("no args => default test/ scope, no selectors", () => {
    expect(selectorsPresent([])).toBe(false);
    expect(buildArgs([])).toEqual(
      ["test", "--timeout", "60000", "--preload", "./src/test-preload.ts", "test/"]
    );
  });

  test("one file selector => no default test/", () => {
    const a = ["test/store.test.ts"];
    expect(selectorsPresent(a)).toBe(true);
    expect(buildArgs(a)).toEqual(
      ["test", "--timeout", "60000", "--preload", "./src/test-preload.ts", "test/store.test.ts"]
    );
  });

  test("two file selectors => both preserved, no default", () => {
    const a = ["test/a.test.ts", "test/b.test.ts"];
    expect(selectorsPresent(a)).toBe(true);
    const args = buildArgs(a);
    expect(args).toContain("test/a.test.ts");
    expect(args).toContain("test/b.test.ts");
    expect(args).not.toContain("test/");
  });

  test("-t only => default scope with filter", () => {
    const a = ["-t", "Reindex"];
    expect(selectorsPresent(a)).toBe(false);
    expect(buildArgs(a)).toEqual(
      ["test", "--timeout", "60000", "--preload", "./src/test-preload.ts", "test/", "-t", "Reindex"]
    );
  });

  test("--test-name-pattern only => default scope", () => {
    const a = ["--test-name-pattern", "package"];
    expect(selectorsPresent(a)).toBe(false);
    expect(buildArgs(a)).toContain("test/");
  });

  test("timeout value + name only => default scope, 60000 not positional", () => {
    const a = ["--timeout", "60000", "-t", "package"];
    expect(selectorsPresent(a)).toBe(false);
    const args = buildArgs(a);
    expect(args).toContain("test/");
    expect(args.indexOf("60000")).toBeGreaterThan(0);
  });

  test("equals forms preserved", () => {
    const a = ["--timeout=60000", "-t=package"];
    expect(selectorsPresent(a)).toBe(false);
    const args = buildArgs(a);
    expect(args).toContain("--timeout=60000");
    expect(args).toContain("-t=package");
  });

  test("explicit file + flags => file takes precedence, no default", () => {
    const a = ["test/package.test.ts", "-t", "package"];
    expect(selectorsPresent(a)).toBe(true);
    const args = buildArgs(a);
    expect(args).toContain("test/package.test.ts");
    expect(args).not.toContain("test/");
  });

  test("-- terminator protects selectors after flags", () => {
    const a = ["--timeout", "10000", "--", "test/store.test.ts"];
    expect(selectorsPresent(a)).toBe(true);
    expect(buildArgs(a)).toContain("test/store.test.ts");
    expect(buildArgs(a)).not.toContain("test/");
  });

  test("optional --bail bare with file => warning path, no steal", () => {
    const a = ["--bail", "test/store.test.ts"];
    expect(selectorsPresent(a)).toBe(true);
  });

  test("spawn failure mapping: error => exit 1", () => {
    expect(childExitCode({ status: null, signal: null, error: new Error("spawn") })).toBe(1);
    expect(childExitCode({ status: 1, signal: null })).toBe(1);
    expect(childExitCode({ status: 0, signal: null })).toBe(0);
    expect(childExitCode({ status: null, signal: "SIGTERM" })).toBe(1);
  });
});
