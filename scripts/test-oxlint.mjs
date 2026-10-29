#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { readdirSync } from "node:fs";
import { join } from "node:path";

const root = fileURLToPath(new URL("..", import.meta.url));

function findFiles(directory = "tools/oxlint") {
  return readdirSync(join(root, directory), { withFileTypes: true }).flatMap(entry => {
    const path = join(directory, entry.name);
    if (entry.isDirectory()) return findFiles(path);
    if (entry.isFile() && entry.name.endsWith(".test.ts")) return [path];
    return [];
  }).sort();
}

const files = findFiles();
if (files.length === 0) throw new Error("No Oxlint rule test modules found");
console.log(`==> Running ${files.length} rule test modules under Node`);

for (const f of files) {
  const label = `Rule test ${f}`;
  console.log(`==> ${label}`);
  const res = spawnSync(process.execPath, ["--import", "tsx", f], {
    cwd: root,
    stdio: "inherit",
    env: { ...process.env },
  });
  if (res.status !== 0) {
    console.error(`Failed: ${label} (exit ${res.status ?? res.signal})`);
    process.exit(res.status ?? 1);
  }
}
console.log(`==> All ${files.length} rule test modules passed`);
