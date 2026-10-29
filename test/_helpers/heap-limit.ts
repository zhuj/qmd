import type { SQLiteValue } from "../../src/db.js";

export interface MinimalDbPort {
  prepare(sql: string): {
    get(...params: SQLiteValue[]): unknown;
  };
}

/**
 * Whether SQLite enforces the hard heap limit just set on `db`: an allocation
 * past it fails with SQLITE_NOMEM only when SQLite tracks memory. A driver built
 * with DEFAULT_MEMSTATUS=0 may not enforce it, irrespective of the JS runtime;
 * a heap-budget test proves its budget only where this returns true.
 */
export function heapLimitBinds(db: MinimalDbPort, limitBytes: number): boolean {
  try {
    db.prepare("SELECT length(randomblob(?)) AS n").get(limitBytes + 8 * 1024 * 1024);
    return false;
  } catch (err) {
    if (err && typeof err === "object" && "code" in err && err.code === "SQLITE_NOMEM") {
      return true;
    }
    throw err;
  }
}

export function verifyHeapLimit(db: MinimalDbPort, limitBytes: number, required: boolean): boolean {
  const binds = heapLimitBinds(db, limitBytes);
  if (required && !binds) {
    throw new Error(
      "memory accounting unavailable; qualified better-sqlite3 build needed (limit set to " +
        limitBytes +
        " but SQLite did not enforce it with SQLITE_NOMEM)"
    );
  }
  return binds;
}
