/**
 * db.ts - Cross-runtime SQLite compatibility layer
 *
 * Uses better-sqlite3 for all SQLite access, regardless of runtime (Bun or Node).
 * Keeps a shared Database / Statement interface for callers on either runtime.
 */

export const isBun = "Bun" in globalThis;

export type SQLiteValue = string | number | bigint | Buffer | Uint8Array | Float32Array | null;
export type SQLiteParams = readonly SQLiteValue[];

type DatabaseConstructor = new (path: string) => Database;
type LoadableSqliteDatabase = Pick<Database, "loadExtension">;

const _Database = (await import("better-sqlite3")).default as DatabaseConstructor;
const sqliteVec = await import("sqlite-vec");
const _sqliteVecLoad = (db: LoadableSqliteDatabase) => sqliteVec.load(db as Parameters<typeof sqliteVec.load>[0]);

function isBusyError(err: unknown): boolean {
  if (typeof err !== "object" || err === null) return false;
  const code = (err as { code?: unknown }).code;
  if (code === "SQLITE_BUSY" || code === "SQLITE_BUSY_SNAPSHOT") return true;
  const message = (err as { message?: unknown }).message;
  return typeof message === "string" && /database is locked|database is busy|SQLITE_BUSY/i.test(message);
}

function sleepSync(ms: number): void {
  Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

/**
 * Switch a connection to WAL, retrying on `SQLITE_BUSY` within the busy-timeout
 * budget. Unlike ordinary writes, migrating the journal needs a brief exclusive
 * lock and does NOT invoke the busy handler, so concurrent first-time opens of a
 * cold database throw "database is locked" even with `busy_timeout` set. Once the
 * database is already WAL the pragma is a cheap no-op that does not contend.
 */
function enableWal(db: Database, budgetMs: number): void {
  const deadline = Date.now() + Math.max(budgetMs, 0);
  for (let attempt = 0; ; attempt++) {
    try {
      db.exec("PRAGMA journal_mode = WAL");
      return;
    } catch (err) {
      if (!isBusyError(err) || Date.now() >= deadline) throw err;
      sleepSync(Math.min(5 + attempt, 25));
    }
  }
}

/**
 * Open a better-sqlite3 database under either Node.js or Bun.
 *
 * Use an explicit busy timeout so concurrent writers wait at batch boundaries
 * rather than exhausting the driver's shorter default. WAL improves
 * read-while-write concurrency but does not serialise writers. Setting the
 * timeout at connection open makes parallel processes (e.g. an `update` or
 * `query` racing a long `embed`, or a first-open schema migration racing any
 * routine command) queue at batch boundaries instead of failing on contact.
 *
 * WAL is enabled here too (with a bounded retry) so connection-level pragmas
 * live in one place and the cold-database journal migration survives concurrent
 * opens.
 *
 * Default 120_000 ms outlasts the worst-case batch commit on a multi-GB
 * index. Override with `QMD_SQLITE_BUSY_TIMEOUT` (value in milliseconds; `0`
 * selects fail-fast behaviour).
 */
export function openDatabase(path: string): Database {
  const db = new _Database(path) as Database;
  const raw = process.env.QMD_SQLITE_BUSY_TIMEOUT;
  const parsed = raw !== undefined && raw !== "" ? Number(raw) : Number.NaN;
  const busyTimeoutMs = Number.isFinite(parsed) && parsed >= 0 ? Math.floor(parsed) : 120_000;
  db.exec(`PRAGMA busy_timeout = ${busyTimeoutMs}`);
  enableWal(db, busyTimeoutMs);
  return db;
}

/**
 * Common subset of the Database interface used throughout QMD.
 */
export interface Database {
  /** True while a transaction or savepoint is open. */
  readonly inTransaction: boolean;
  exec(sql: string): void;
  prepare(sql: string): Statement;
  loadExtension(path: string): void;
  // better-sqlite3 returns the wrapped function with transaction variants attached.
  transaction<T extends (...args: SQLiteValue[]) => unknown>(fn: T): T & { immediate: T };
  close(): void;
}

export interface Statement {
  run(...params: SQLiteValue[]): { changes: number; lastInsertRowid: number | bigint };
  get<T = unknown>(...params: SQLiteValue[]): T | undefined;
  all<T = unknown>(...params: SQLiteValue[]): T[];
  iterate<T = unknown>(...params: SQLiteValue[]): IterableIterator<T>;
}

/**
 * Load the sqlite-vec extension into a database.
 *
 * Propagates extension-loading failures to the caller.
 */
export function loadSqliteVec(db: Database): void {
  _sqliteVecLoad(db);
}
