# SQLite Memory Testing

QMD uses `better-sqlite3` under both Node.js and Bun. SQLite heap-limit enforcement depends on the native SQLite build and initialization, not on the JavaScript runtime.

## Two Test Tiers

### Functional Tests

```sh
CI=true bun run test
```

The normal suite always checks sampling, exact fake-vector reproduction, legacy adoption and duplicate-document handling. The memory-related workers also probe the requested SQLite hard heap limit and report `limitBinds` and their actual JavaScript runtime.

A successful functional test does not prove a memory budget when `limitBinds` is `false`. These tests do not skip functionality when accounting is unavailable. Their document sizes, duplicate-path counts and requested limits remain unchanged.

### Strict SQLite Memory Qualification

```sh
CI=true bun run test:sqlite-memory
```

This separate gate requires all four worker runs:

| Operation | SQLite Hard Heap Limit | Runtimes |
| --- | --- | --- |
| Doctor vector reproduction with duplicate document paths | 32 MiB | Node.js and Bun |
| Legacy adoption of one body shared by 200 active paths | 16 MiB | Node.js and Bun |

The runner forces `QMD_TEST_SQLITE_MEMORY_STRICT=1` for its children. Workers require enforcement after fixture setup and before the operation being qualified. A missing runtime, unavailable enforcement, failed operation or invalid result fails qualification. Neither runtime is substituted for the other.

The probe deliberately requests an allocation above the configured limit. Only an error with code `SQLITE_NOMEM` establishes enforcement; unrelated errors propagate instead of masquerading as a successful probe.

Strict qualification is separate from the default functional suite. A green default suite must not be reported as green memory qualification or full feature acceptance.

## Provisioning a Qualified Native Build

Use an isolated qualification environment with the project's locked dependency versions and a reviewed `better-sqlite3` native artifact. Keep standard production installations independent of this test-only provisioning.

The installed package's build recipe currently specifies `SQLITE_DEFAULT_MEMSTATUS=0`. A memory-enabled qualification recipe must replace that setting, for example with `SQLITE_DEFAULT_MEMSTATUS=1`, without leaving conflicting definitions. Memory accounting must remain enabled during SQLite initialization. Other SQLite allocator or page-cache configurations can also affect enforcement, so a compile flag alone is not qualification evidence: the strict gate must pass on the actual artifact under both runtimes.

Record the dependency versions, native artifact provenance, SQLite version, build configuration, Node/Bun versions, OS/architecture and strict-gate output for a qualified environment. The 16 MiB and 32 MiB limits cover SQLite-managed memory, not total process RSS, JavaScript heaps or model memory.

No native dependency is rebuilt, downloaded or replaced by either test command. Provisioning a different artifact requires separately authorized build/install work. Until the strict gate has passed against such an artifact, memory qualification remains blocked even when functional tests pass.
