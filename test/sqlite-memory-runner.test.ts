import { describe, expect, test } from "vitest";
import {
  buildSqliteMemoryJobs,
  validateSqliteMemoryResult,
  runSqliteMemoryJob,
} from "../scripts/test-sqlite-memory.mjs";

describe("sqlite-memory runner (fake/injected)", () => {
  const root = process.cwd();

  function makeFakeSpawn(resultObj: Record<string, unknown>, exitStatus = 0, stderr = "") {
    return (cmd: string, args: string[], opts: { cwd?: string; shell?: boolean; env?: Record<string, string | undefined>; stdio?: unknown; timeout?: number }) => {
      expect(opts).toBeDefined();
      expect(opts.shell).toBe(false);
      expect(opts.cwd).toBe(root);
      expect(opts.env).toBeDefined();
      expect(opts.env?.QMD_TEST_SQLITE_MEMORY_STRICT).toBe("1");
      expect(opts.env?.CI).toBe("true");
      expect(opts.timeout).toBe(60000);
      return {
        status: exitStatus,
        signal: null,
        stdout: Buffer.from(JSON.stringify(resultObj)),
        stderr: Buffer.from(stderr),
        error: undefined,
      };
    };
  }

  test("build produces exactly 4 jobs with 2 budgets and 2 runtimes", () => {
    const jobs = buildSqliteMemoryJobs();
    expect(jobs).toHaveLength(4);
    const nodes = jobs.filter((j) => j.cmd === process.execPath);
    const buns = jobs.filter((j) => j.cmd === "bun");
    expect(nodes).toHaveLength(2);
    expect(buns).toHaveLength(2);
    const budgets = new Set(jobs.map((j) => j.budget));
    expect(budgets.has(33554432)).toBe(true);
    expect(budgets.has(16777216)).toBe(true);
  });

  test("forced strict flag overrides caller 0", () => {
    const jobs = buildSqliteMemoryJobs();
    const fake = makeFakeSpawn({ ok: true, details: "3 sampled chunks reproduce stored vectors", limitBinds: true, runtime: "node", budgetBytes: 33554432 });
    const original = process.env.QMD_TEST_SQLITE_MEMORY_STRICT;
    process.env.QMD_TEST_SQLITE_MEMORY_STRICT = "0";
    try {
      const res = runSqliteMemoryJob(jobs[0], fake);
      expect(res.result.limitBinds).toBe(true);
      expect(process.env.QMD_TEST_SQLITE_MEMORY_STRICT).toBe("0");
    } finally {
      if (original === undefined) delete process.env.QMD_TEST_SQLITE_MEMORY_STRICT;
      else process.env.QMD_TEST_SQLITE_MEMORY_STRICT = original;
    }
  });

  test("all 4 valid results green", () => {
    const jobs = buildSqliteMemoryJobs();
    const validDoctor = { ok: true, details: "3 sampled chunks reproduce stored vectors", limitBinds: true, runtime: "node", budgetBytes: 33554432 };
    const validLegacy = { checked: true, adopted: 1, limitBinds: true, runtime: "bun", budgetBytes: 16777216 };
    const results = [
      runSqliteMemoryJob(jobs[0], makeFakeSpawn({ ...validDoctor, runtime: "node" })),
      runSqliteMemoryJob(jobs[1], makeFakeSpawn({ ...validDoctor, runtime: "bun" })),
      runSqliteMemoryJob(jobs[2], makeFakeSpawn({ ...validLegacy, runtime: "node" })),
      runSqliteMemoryJob(jobs[3], makeFakeSpawn({ ...validLegacy, runtime: "bun" })),
    ];
    expect(results).toHaveLength(4);
    expect(results.every((r) => r.result.limitBinds === true)).toBe(true);
  });

  test("success but limitBinds false FAIL", () => {
    const jobs = buildSqliteMemoryJobs();
    const fake = makeFakeSpawn({ ok: true, details: "x", limitBinds: false, runtime: "node" });
    expect(() => runSqliteMemoryJob(jobs[0], fake)).toThrow(/limitBinds !== true/);
  });

  test("wrong budget FAIL", () => {
    const jobs = buildSqliteMemoryJobs();
    const fake = makeFakeSpawn({ ok: true, details: "3 sampled chunks reproduce stored vectors", limitBinds: true, runtime: "node", budgetBytes: 999 });
    expect(() => runSqliteMemoryJob(jobs[0], fake)).toThrow(/wrong budgetBytes/);
  });

  test("wrong result details FAIL", () => {
    const jobs = buildSqliteMemoryJobs();
    const fake = makeFakeSpawn({ ok: true, details: "not 3 sampled chunks reproduce stored vectors", limitBinds: true, runtime: "node", budgetBytes: 33554432 });
    expect(() => runSqliteMemoryJob(jobs[0], fake)).toThrow(/unexpected operation result/);
  });

  test("label rename cannot bypass operation checks", () => {
    const jobs = buildSqliteMemoryJobs();
    const renamed = { ...jobs[0], label: "renamed" };
    const fake = makeFakeSpawn({ ok: true, details: "3 sampled chunks reproduce stored vectors", limitBinds: true, runtime: "node", budgetBytes: 33554432 });
    expect(runSqliteMemoryJob(renamed, fake)).toBeDefined();
    const badFake = makeFakeSpawn({ ok: false, details: "3 sampled chunks reproduce stored vectors", limitBinds: true, runtime: "node", budgetBytes: 33554432 });
    expect(() => runSqliteMemoryJob(renamed, badFake)).toThrow(/unexpected operation result/);
  });

  test("successful stderr FAIL", () => {
    const jobs = buildSqliteMemoryJobs();
    const fake = makeFakeSpawn({ ok: true, details: "3 sampled chunks reproduce stored vectors", limitBinds: true, runtime: "node", budgetBytes: 33554432 }, 0, "unexpected");
    expect(() => runSqliteMemoryJob(jobs[0], fake)).toThrow(/unexpected child stderr/);
  });

  test("nonzero exit preserves diagnostic stderr", () => {
    const jobs = buildSqliteMemoryJobs();
    const fake = makeFakeSpawn({ ok: true, details: "3 sampled chunks reproduce stored vectors", limitBinds: true, runtime: "node", budgetBytes: 33554432 }, 1, "memory accounting unavailable; qualified better-sqlite3 build needed");
    expect(() => runSqliteMemoryJob(jobs[0], fake)).toThrow(/memory accounting unavailable/);
  });

  test("wrong runtime FAIL", () => {
    const jobs = buildSqliteMemoryJobs();
    const fake = makeFakeSpawn({ ok: true, details: "3 sampled chunks reproduce stored vectors", limitBinds: true, runtime: "wrong", budgetBytes: 33554432 });
    expect(() => runSqliteMemoryJob(jobs[0], fake)).toThrow(/wrong runtime/);
  });

  test("malformed JSON FAIL", () => {
    const jobs = buildSqliteMemoryJobs();
    const fake = () => ({ status: 0, signal: null, stdout: Buffer.from("not json"), stderr: Buffer.from(""), error: undefined });
    expect(() => runSqliteMemoryJob(jobs[0], fake)).toThrow(/malformed JSON/);
  });

  test("missing fields FAIL", () => {
    const jobs = buildSqliteMemoryJobs();
    const fake = makeFakeSpawn({ limitBinds: true, runtime: "node", budgetBytes: 33554432 });
    expect(() => runSqliteMemoryJob(jobs[0], fake)).toThrow();
  });

  test("nonzero exit FAIL", () => {
    const jobs = buildSqliteMemoryJobs();
    const fake = makeFakeSpawn({ ok: true, details: "3 sampled chunks reproduce stored vectors", limitBinds: true, runtime: "node", budgetBytes: 33554432 }, 1);
    expect(() => runSqliteMemoryJob(jobs[0], fake)).toThrow(/exit 1/);
  });

  test("null status FAIL", () => {
    const jobs = buildSqliteMemoryJobs();
    const fake = () => ({ status: null, signal: null, stdout: Buffer.from('{"ok":true,"limitBinds":true,"runtime":"node","budgetBytes":33554432,"details":"3 sampled chunks reproduce stored vectors"}'), stderr: Buffer.from(""), error: undefined });
    expect(() => runSqliteMemoryJob(jobs[0], fake)).toThrow(/exit null/);
  });

  test("spawn error FAIL", () => {
    const jobs = buildSqliteMemoryJobs();
    const fake = () => ({ status: 0, signal: null, stdout: Buffer.from("{}"), stderr: Buffer.from(""), error: new Error("ENOENT") });
    expect(() => runSqliteMemoryJob(jobs[0], fake)).toThrow(/spawn error/);
  });

  test("missing Bun executable FAIL (no skip)", () => {
    const jobs = buildSqliteMemoryJobs();
    const bunJob = jobs.find((j) => j.cmd === "bun" && j.label.includes("doctor"))!;
    const fake = () => ({ status: 0, signal: null, stdout: Buffer.from("{}"), stderr: Buffer.from(""), error: new Error("ENOENT: bun") });
    expect(() => runSqliteMemoryJob(bunJob, fake)).toThrow(/spawn error/);
  });

  test("dispatches the selected job exactly once with its configured arguments", () => {
    const job = buildSqliteMemoryJobs()[0];
    const fake = makeFakeSpawn({ ...job.expectedFields, limitBinds: true, runtime: job.expectedRuntime, budgetBytes: job.budget });
    let calls = 0;
    runSqliteMemoryJob(job, (cmd, args, opts) => {
      calls++;
      expect(cmd).toBe(job.cmd);
      expect(args).toEqual(job.args);
      return fake(cmd, args, opts);
    });
    expect(calls).toBe(1);
  });

  test("validateSqliteMemoryResult throws descriptive for unsupported limit", () => {
    const job = buildSqliteMemoryJobs()[0];
    const okResult = { ok: true, limitBinds: false, runtime: "node", details: "x", budgetBytes: 33554432 };
    expect(() => validateSqliteMemoryResult(okResult, job)).toThrow(/limitBinds !== true/);
  });
});
