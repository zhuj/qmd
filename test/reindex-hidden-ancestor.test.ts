import { describe, test, expect } from "vitest";
import { mkdtemp, mkdir, writeFile, rm, symlink, realpath } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { createStore, reindexCollection } from "../src/store.js";

describe("reindex hidden ancestor filtering", () => {
  test("symlink alias root: physical hidden not indexed; visible indexed; outside reported", async () => {
    const root = await mkdtemp(join(tmpdir(), "qmd-reindex-alias-"));
    let store: ReturnType<typeof createStore> | undefined;
    try {
      const realCol = join(root, "realcollection");
      await mkdir(realCol, { recursive: true });
      const physicalCol = await realpath(realCol);
      const alias = join(root, "alias");
      await symlink(physicalCol, alias, "junction");
      await writeFile(join(realCol, "visible.md"), "# visible\nalias-safe\n");
      await writeFile(join(realCol, ".hidden.md"), "# hidden\nbad\n");
      const outsideDir = join(root, "outside");
      await mkdir(outsideDir, { recursive: true });
      await writeFile(join(outsideDir, "escape.md"), "# escape\nbad\n");
      const dbPath = join(root, "test.sqlite");
      store = createStore(dbPath);
      await reindexCollection(store, alias, join(physicalCol, "visible.md"), "alias-visible");
      const visBodies = store.db.prepare(`SELECT content.doc AS body FROM documents d JOIN content ON content.hash = d.hash WHERE d.collection = ? AND d.active = 1`).all("alias-visible") as { body: string }[];
      expect(visBodies.some(b => b.body.includes("alias-safe"))).toBe(true);
      const hidden = await reindexCollection(store, alias, join(physicalCol, ".hidden.md"), "alias-hidden");
      expect(hidden.indexed).toBe(0);
      const hiddenBodies = store.db.prepare(`SELECT content.doc AS body FROM documents d JOIN content ON content.hash = d.hash WHERE d.collection = ? AND d.active = 1`).all("alias-hidden") as { body: string }[];
      expect(hiddenBodies.length).toBe(0);
      const out = await reindexCollection(store, alias, join(outsideDir, "escape.md"), "alias-out");
      expect(out.skippedFiles.some(s => s.code === "OUTSIDE_COLLECTION")).toBe(true);
      const outBodies = store.db.prepare(`SELECT content.doc AS body FROM documents d JOIN content ON content.hash = d.hash WHERE d.collection = ? AND d.active = 1`).all("alias-out") as { body: string }[];
      expect(outBodies.length).toBe(0);
    } finally {
      try { store?.close(); } finally { await rm(root, { recursive: true, force: true }); }
    }
  });

  test("collection under hidden ancestor; absolute inside indexed; outside reported; hidden within excluded", async () => {
    const root = await mkdtemp(join(tmpdir(), "qmd-reindex-hidden-ancestor-"));
    let store: ReturnType<typeof createStore> | undefined;
    try {
    const ancestor = join(root, ".ancestor");
    await mkdir(ancestor, { recursive: true });
    const collection = join(ancestor, "collection");
    await mkdir(collection, { recursive: true });

    await writeFile(join(collection, "inside.md"), "# inside\nsafe\n");
    await writeFile(join(collection, ".hidden.md"), "# hidden\nbad\n");
    await mkdir(join(collection, ".hiddendir"), { recursive: true });
    await writeFile(join(collection, ".hiddendir", "bad.md"), "# hidden-dir\nbad\n");

    const outsideDir = join(root, "outside");
    await mkdir(outsideDir, { recursive: true });
    await writeFile(join(outsideDir, "escape.md"), "# escape\nbad\n");

    const dbPath = join(root, "test.sqlite");
    store = createStore(dbPath);
      // Relative glob under hidden ancestor
      const rel = await reindexCollection(store, collection, "**/*.md", "docs");
      const bodies = store.db.prepare(`
        SELECT content.doc AS body FROM documents d
        JOIN content ON content.hash = d.hash
        WHERE d.collection = ? AND d.active = 1
      `).all("docs") as { body: string }[];

      expect(bodies.some(b => b.body.includes("safe"))).toBe(true);
      expect(bodies.map(b => b.body).join("")).not.toContain("bad");
      expect(rel.indexed).toBeGreaterThanOrEqual(1);

      // Absolute path to file inside collection (non-hidden) must index
      await reindexCollection(store, collection, join(collection, "inside.md"), "docs2");
      const absBodies = store.db.prepare(`
        SELECT content.doc AS body FROM documents d
        JOIN content ON content.hash = d.hash
        WHERE d.collection = ? AND d.active = 1
      `).all("docs2") as { body: string }[];
      expect(absBodies.some(b => b.body.includes("safe"))).toBe(true);

      const hiddenFile = await reindexCollection(store, collection, join(collection, ".hidden.md"), "hidden-file");
      const hiddenDirectory = await reindexCollection(store, collection, join(collection, ".hiddendir", "bad.md"), "hidden-directory");
      expect(hiddenFile.indexed).toBe(0);
      expect(hiddenDirectory.indexed).toBe(0);

      // Absolute path outside collection must be reported OUTSIDE_COLLECTION
      // and its body never indexed.
      const outsideAbs = await reindexCollection(store, collection, join(outsideDir, "escape.md"), "docs3");
      expect(outsideAbs.skippedFiles.some(s => s.code === "OUTSIDE_COLLECTION")).toBe(true);
      const outBodies = store.db.prepare(`
        SELECT content.doc AS body FROM documents d
        JOIN content ON content.hash = d.hash
        WHERE d.collection = ? AND d.active = 1
      `).all("docs3") as { body: string }[];
      expect(outBodies.length).toBe(0);
      expect(outBodies.map(b => b.body).join("")).not.toContain("bad");
      const parentRelative = await reindexCollection(store, collection, "../../outside/escape.md", "parent-relative");
      expect(parentRelative.skippedFiles.some(s => s.code === "OUTSIDE_COLLECTION")).toBe(true);
    } finally {
      try { store?.close(); }
      finally { await rm(root, { recursive: true, force: true }); }
    }
  });
});
