import { expect, test, describe } from "vitest";
import { nativeFixture } from "./_helpers/onnx-native.js";
import { createNativeEmbeddingSession } from "../src/embeddings/onnx-native-session.js";

describe("T7B1-01 admission-regression evidence", () => {
  const fixtureError = () => Object.assign(new Error("fixture private detail"), {
    cause: new Error("fixture cause"), path: "fixture-only-path",
  });

  test("supported layout constructor and close", async () => {
    const f = nativeFixture();
    const session = createNativeEmbeddingSession(f.objects);
    expect(typeof session.run).toBe("function");
    expect(typeof session.tokenizer.encode).toBe("function");
    expect(typeof session.tokenizer.decode).toBe("function");
    expect(f.modelCloses()).toBe(0); expect(f.forwardCalls()).toBe(0); expect(f.feeds.length).toBe(0); expect(f.calls.length).toBe(0);
    await session.close();
    expect(f.modelCloses()).toBe(1);
  });

  function assertAdmissionFailure(f: ReturnType<typeof nativeFixture>, caught: unknown) {
    expect(caught).toBeInstanceOf(Error);
    if (!(caught instanceof Error)) throw new Error("not Error");
    expect(caught.message).toBe("Unsupported ONNX session contract");
    expect("cause" in caught).toBe(false);
    expect("path" in caught).toBe(false);
    expect(f.feeds.length).toBe(0); expect(f.calls.length).toBe(0);
    expect(f.modelCloses()).toBe(0); expect(f.forwardCalls()).toBe(0);
    expect(caught.message).not.toContain("fixture");
    expect(caught.message).not.toContain("private");
    expect(caught.message).not.toContain("fixture-only");
  }

  test("rejects model.sessions getter exception", () => {
    const f = nativeFixture();
    Object.defineProperty(f.model, "sessions", { get() { throw fixtureError(); } });
    let caught: unknown;
    try { createNativeEmbeddingSession(f.objects); } catch (e) { caught = e; }
    assertAdmissionFailure(f, caught);
  });

  test("rejects raw session inputNames getter exception", () => {
    const f = nativeFixture();
    Object.defineProperty(f.raw, "inputNames", { get() { throw fixtureError(); } });
    let caught: unknown;
    try { createNativeEmbeddingSession(f.objects); } catch (e) { caught = e; }
    assertAdmissionFailure(f, caught);
  });

  test("rejects tokenizer.encode getter exception", () => {
    const f = nativeFixture();
    const tok = f.objects.tokenizer;
    if (typeof tok !== "function") throw new Error("fixture tokenizer missing");
    Object.defineProperty(tok, "encode", { get() { throw fixtureError(); } });
    let caught: unknown;
    try { createNativeEmbeddingSession(f.objects); } catch (e) { caught = e; }
    assertAdmissionFailure(f, caught);
  });
});
