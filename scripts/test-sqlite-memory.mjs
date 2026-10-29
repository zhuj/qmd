#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

/** @typedef {Object} JobConfig
 * @property {string} label
 * @property {string} cmd
 * @property {string[]} args
 * @property {number} budget
 * @property {string} expectedRuntime
 * @property {Record<string,unknown>} expectedFields
 */

const ROOT = resolve(fileURLToPath(new URL(".", import.meta.url)), "..");
const TSX = join(ROOT, "node_modules", "tsx", "dist", "cli.mjs");
const DOCTOR = join(ROOT, "test", "_helpers", "doctor-vector-check-worker.ts");
const LEGACY = join(ROOT, "test", "_helpers", "legacy-adoption-shared-body-worker.ts");

/** @returns {[JobConfig, JobConfig, JobConfig, JobConfig]} */
export function buildSqliteMemoryJobs() {
  return [
    {
      label: "doctor-vector-check (node)", cmd: process.execPath, args: [TSX, DOCTOR],
      budget: 33554432, expectedRuntime: "node",
      expectedFields: { ok: true, details: "3 sampled chunks reproduce stored vectors" },
    },
    {
      label: "doctor-vector-check (bun)", cmd: "bun", args: [DOCTOR],
      budget: 33554432, expectedRuntime: "bun",
      expectedFields: { ok: true, details: "3 sampled chunks reproduce stored vectors" },
    },
    {
      label: "legacy-adoption-shared-body (node)", cmd: process.execPath, args: [TSX, LEGACY],
      budget: 16777216, expectedRuntime: "node",
      expectedFields: { checked: true, adopted: 1 },
    },
    {
      label: "legacy-adoption-shared-body (bun)", cmd: "bun", args: [LEGACY],
      budget: 16777216, expectedRuntime: "bun",
      expectedFields: { checked: true, adopted: 1 },
    },
  ];
}

/** @param {unknown} result @param {JobConfig} job @returns {Record<string, unknown>} */
export function validateSqliteMemoryResult(result, job) {
  if (result === null || typeof result !== "object") {
    throw new Error(`${job.label}: result is not an object (got ${String(result)})`);
  }
  const r = /** @type {Record<string,unknown>} */ (result);
  if (r.limitBinds !== true) {
    throw new Error(`${job.label}: limitBinds !== true (got ${r.limitBinds}); strict SQLite memory qualification failed`);
  }
  if (r.runtime !== job.expectedRuntime) {
    throw new Error(`${job.label}: wrong runtime field (got ${r.runtime}, expected ${job.expectedRuntime})`);
  }
  if (r.budgetBytes !== job.budget) {
    throw new Error(`${job.label}: wrong budgetBytes (got ${r.budgetBytes}, expected ${job.budget})`);
  }
  for (const [k, v] of Object.entries(job.expectedFields)) {
    if (r[k] !== v) {
      throw new Error(`${job.label}: unexpected operation result ${k}=${r[k]} (expected ${v})`);
    }
  }
  return r;
}

/** @typedef {{cwd:string,shell:false,env:NodeJS.ProcessEnv,stdio:"pipe",timeout:number,encoding:"buffer"}} WorkerOptions */
/** @typedef {{status:number|null,signal:NodeJS.Signals|null,stdout:Buffer,stderr:Buffer,error?:Error}} WorkerResult */
/** @param {JobConfig} job @param {(cmd:string,args:string[],opts:WorkerOptions)=>WorkerResult} [spawnImpl] */
export function runSqliteMemoryJob(job, spawnImpl) {
  const spawn = spawnImpl ?? spawnSync;
  const result = spawn(job.cmd, job.args, {
    cwd: ROOT,
    shell: false,
    env: { ...process.env, QMD_TEST_SQLITE_MEMORY_STRICT: "1", CI: "true" },
    stdio: "pipe",
    encoding: "buffer",
    timeout: 60000,
  });
  if (result.error) {
    throw new Error(`${job.label}: spawn error (${result.error.message})`);
  }
  if (result.status === null || result.status !== 0) {
    const stderrText = result.stderr ? result.stderr.toString("utf8") : "";
    const bounded = stderrText.length > 2000 ? stderrText.slice(0, 2000) + "...(truncated)" : stderrText;
    const sig = result.signal ? `signal=${result.signal}` : "no signal";
    throw new Error(`${job.label}: exit ${result.status} (${sig}) stderr=${bounded}`);
  }
  const stdoutText = result.stdout ? result.stdout.toString("utf8") : "";
  const stderrText = result.stderr ? result.stderr.toString("utf8") : "";
  if (stderrText.trim().length > 0) {
    throw new Error(`${job.label}: unexpected child stderr=${stderrText.slice(0, 2000)}`);
  }
  const trimmed = stdoutText.trim();
  if (!trimmed) {
    throw new Error(`${job.label}: empty stdout`);
  }
  /** @type {unknown} */
  let parsed;
  try {
    parsed = JSON.parse(trimmed);
  } catch (e) {
    const msg = (e && typeof e === "object" && "message" in e) ? String(e.message) : String(e);
    throw new Error(`${job.label}: malformed JSON stdout (${msg}); stdout=${trimmed.slice(0, 200)}`);
  }
  const validated = validateSqliteMemoryResult(parsed, job);
  return { job, result: validated, stdout: stdoutText, stderr: stderrText };
}

function main() {
  if (process.versions.bun) {
    console.error("Run with Node: node scripts/test-sqlite-memory.mjs (or bun run test:sqlite-memory).");
    process.exit(1);
  }
  const jobs = buildSqliteMemoryJobs();
  const results = [];
  let failed = false;
  for (const job of jobs) {
    try {
      const res = runSqliteMemoryJob(job);
      results.push({ label: job.label, ok: true, budget: job.budget, expectedRuntime: job.expectedRuntime, runtime: res.result.runtime, limitBinds: res.result.limitBinds });
    } catch (e) {
      failed = true;
      const msg = (e && typeof e === "object" && "message" in e) ? String(e.message) : String(e);
      results.push({ label: job.label, ok: false, budget: job.budget, expectedRuntime: job.expectedRuntime, error: msg });
      console.error(`FAIL ${job.label}: ${msg}`);
    }
  }
  const passed = results.filter((r) => r.ok).length;
  for (const r of results) {
    console.log(`${r.ok ? "PASS" : "FAIL"} ${r.label} budget=${r.budget} expectedRuntime=${r.expectedRuntime}${r.ok ? ` runtime=${r.runtime} limitBinds=${r.limitBinds}` : ` error=${r.error}`}`);
  }
  console.log(`Aggregate: ${passed}/${results.length}`);
  if (failed || passed !== results.length) {
    process.exit(1);
  }
}

const entryUrl = process.argv[1] ? pathToFileURL(resolve(process.argv[1])) : null;
if (entryUrl && import.meta.url === entryUrl.href) {
  main();
}
