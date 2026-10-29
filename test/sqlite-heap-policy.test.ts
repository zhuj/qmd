import { describe, expect, test } from "vitest";
import { heapLimitBinds, verifyHeapLimit, type MinimalDbPort } from "./_helpers/heap-limit.js";
import type { SQLiteValue } from "../src/db.js";

function fakeDb(result: unknown | (() => unknown)): MinimalDbPort {
  return {
    prepare() {
      return {
        get(..._params: SQLiteValue[]): unknown {
          if (typeof result === "function") return result();
          return result;
        },
      };
    },
  };
}

describe("heap-limit policy (pure, no native qualification)", () => {
  test("successful oversize probe returns false (limit does not bind)", () => {
    const db = fakeDb({ n: 123 });
    expect(heapLimitBinds(db, 100)).toBe(false);
  });

  test("SQLITE_NOMEM object error returns true", () => {
    const err = Object.assign(new Error("out of memory"), { code: "SQLITE_NOMEM" });
    const db = fakeDb(() => { throw err; });
    expect(heapLimitBinds(db, 100)).toBe(true);
  });

  test("SQLITE_ERROR propagates instead of establishing enforcement", () => {
    const err = Object.assign(new Error("table missing"), { code: "SQLITE_ERROR" });
    const db = fakeDb(() => { throw err; });
    expect(() => heapLimitBinds(db, 100)).toThrow(err);
  });

  test("unexpected Error propagates unchanged (not caught as true)", () => {
    const err = new Error("unexpected");
    const db = fakeDb(() => { throw err; });
    expect(() => heapLimitBinds(db, 100)).toThrow(err);
  });

  test("primitive throw propagates unchanged", () => {
    const db = fakeDb(() => { throw 42; });
    try {
      heapLimitBinds(db, 100);
      expect.fail("should throw");
    } catch (e) {
      expect(e).toBe(42);
    }
  });

  test("verifyHeapLimit required false unsupported returns false", () => {
    const db = fakeDb({ n: 1 });
    expect(verifyHeapLimit(db, 100, false)).toBe(false);
  });

  test("verifyHeapLimit required true unsupported throws descriptive Error", () => {
    const db = fakeDb({ n: 1 });
    expect(() => verifyHeapLimit(db, 100, true)).toThrow(/memory accounting unavailable.*qualified better-sqlite3/);
  });

  test("verifyHeapLimit required true SQLITE_NOMEM returns true", () => {
    const err = Object.assign(new Error("oom"), { code: "SQLITE_NOMEM" });
    const db = fakeDb(() => { throw err; });
    expect(verifyHeapLimit(db, 100, true)).toBe(true);
  });

  test("verifyHeapLimit propagates unexpected error in strict mode", () => {
    const err = new Error("other");
    const db = fakeDb(() => { throw err; });
    expect(() => verifyHeapLimit(db, 100, true)).toThrow(err);
  });

  test("assert exact same error object on SQLITE_ERROR propagation", () => {
    const err = Object.assign(new Error("bad"), { code: "SQLITE_ERROR" });
    const db = fakeDb(() => { throw err; });
    try {
      heapLimitBinds(db, 100);
      expect.fail("should throw");
    } catch (e) {
      expect(e).toBe(err);
    }
  });

  test.each([16777216, 33554432])("probes an allocation above the %i-byte limit", limit => {
    let seenSql = "";
    let seenParams: SQLiteValue[] = [];
    const db: MinimalDbPort = {
      prepare(sql) {
        seenSql = sql;
        return { get(...params) { seenParams = params; return undefined; } };
      },
    };
    expect(heapLimitBinds(db, limit)).toBe(false);
    expect(seenSql).toBe("SELECT length(randomblob(?)) AS n");
    expect(seenParams).toEqual([limit + 8 * 1024 * 1024]);
  });
});
