#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = fileURLToPath(new URL("..", import.meta.url));

// Flags that always require a value (next token or =val).
const valFlags = new Set([
  "--timeout", "-t", "--test-name-pattern",
  "--rerun-each", "--retry", "--seed",
  "--coverage-reporter", "--coverage-dir",
  "--reporter", "--reporter-outfile",
  "--max-concurrency", "--path-ignore-patterns",
  "--shard", "--timings", "--parallel-delay",
  "--parallel",
]);

// Optional-value flags: must use =val to pass value; bare form uses default and never consumes selector.
const optFlags = new Set(["--bail", "--changed"]);

function selectorsPresent(args) {
  for (let i = 0; i < args.length; i++) {
    const a = args[i];
    if (a === "--") {
      // Everything after -- is selector; at least one selector if anything remains.
      return i + 1 < args.length;
    }
    if (!a.startsWith("-")) {
      return true; // positional file pattern
    }
    // Flag form
    if (a.includes("=")) {
      // =val: flag part is left of =; value is right; no extra token consumed.
      continue;
    }
    const flag = a;
    if (valFlags.has(flag)) {
      i++; // consume required value token
      continue;
    }
    if (optFlags.has(flag)) {
      // Bare optional: do NOT consume next token; it remains potential selector.
      // Diagnostic for user clarity when a non-flag follows.
      if (i + 1 < args.length && !args[i + 1].startsWith("-")) {
        console.warn(
          `Note: ${flag} without =val uses default; use ${flag}=N to pass value explicitly`
        );
      }
      continue;
    }
    // Other flags (--describe, --dots, --only, etc.) take no value.
    continue;
  }
  return false;
}

export { selectorsPresent };

function buildArgs(extra) {
  const hasSelector = selectorsPresent(extra);
  const args = ["test", "--timeout", "60000", "--preload", "./src/test-preload.ts"];
  if (!hasSelector) args.push("test/");
  args.push(...extra);
  return args;
}
export { buildArgs };

export function childExitCode(result) {
  return result.error || result.signal ? 1 : result.status ?? 1;
}

const isMain = import.meta.url === pathToFileURL(process.argv[1] || "").href;
if (isMain) {
  const extra = process.argv.slice(2);
  const args = buildArgs(extra);

  console.log(`==> bun ${args.join(" ")}`);
const res = spawnSync("bun", args, {
  cwd: root,
  stdio: "inherit",
  shell: false,
});
if (res.error) {
  console.error(`Failed to spawn bun: ${res.error.message || res.error}`);
  process.exit(1);
}
if (res.signal) {
  console.error(`bun terminated by signal ${res.signal}`);
  process.exit(1);
}
process.exit(childExitCode(res));
}
