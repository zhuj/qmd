import { expect, test } from "vitest";
import { Agent } from "node:https";
import type { RequestOptions } from "node:https";
import { createArtifactHttpTransport } from "../src/embeddings/artifact-http.js";
import type { ArtifactHttpRuntime, ArtifactIncoming } from "../src/embeddings/artifact-http.js";
import { deferred } from "./_helpers/deferred.js";

function fixture(settings: { chunks?: readonly unknown[]; complete?: boolean; failStream?: boolean } = {}) {
  let responseCloses = 0, requestCloses = 0, agentCloses = 0, timerClears = 0, starts = 0;
  let expire: (() => void) | undefined, requestError: ((error: Error) => void) | undefined;
  let seenOptions: RequestOptions | undefined;
  const response: ArtifactIncoming = {
    statusCode: 200, rawHeaders: ["Content-Type", "application/octet-stream"], complete: settings.complete ?? true,
    on: () => {}, destroy: () => { responseCloses++; },
    async *[Symbol.asyncIterator]() {
      for (const chunk of settings.chunks ?? [Buffer.from("bytes")]) yield chunk;
      if (settings.failStream) throw new Error("fixture-only private stream detail");
    },
  };
  const runtime: ArtifactHttpRuntime = {
    resolve: async () => [{ address: "8.8.8.8", family: 4 }],
    schedule: (callback, milliseconds) => { expect(milliseconds).toBe(300_000); expire = callback; return () => { timerClears++; }; },
    start: (_url, options, deliver) => {
      starts++; seenOptions = options;
      const agent = options.agent;
      if (!(agent instanceof Agent)) throw new Error("Expected dedicated HTTPS agent");
      const destroy = agent.destroy.bind(agent);
      agent.destroy = () => { agentCloses++; destroy(); };
      return { on: (_event, listener) => { requestError = listener; },
        end: () => queueMicrotask(() => deliver(response)), destroy: () => { requestCloses++; } };
    },
  };
  return { runtime, response, options: () => seenOptions, expire: () => expire?.(),
    error: () => requestError?.(new Error("fixture-only private connection detail")),
    counts: () => ({ responseCloses, requestCloses, agentCloses, timerClears, starts }) };
}
async function consume(chunks: AsyncIterable<Uint8Array>) {
  const bytes: Uint8Array[] = [];
  for await (const chunk of chunks) bytes.push(chunk);
  return Buffer.concat(bytes).toString();
}
const url = () => new URL("https://huggingface.co/fixture/encoder/resolve/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/onnx/model.onnx");
test("uses a dedicated DNS-bound agent while preserving TLS hostname and binary bytes", async () => {
  const f = fixture();
  const reply = await createArtifactHttpTransport(f.runtime)({ url: url(), method: "GET" });
  const options = f.options();
  expect(options?.servername).toBe("huggingface.co");
  expect(options?.rejectUnauthorized).toBe(true);
  expect(options?.maxHeaderSize).toBe(16384);
  expect(options?.headers).toEqual({ "Accept-Encoding": "identity" });
  const lookup = options?.lookup;
  if (lookup === undefined) throw new Error("Missing actual connection lookup");
  const address = await new Promise((resolve) => lookup("huggingface.co", {}, (_error, resolved) => resolve(resolved)));
  expect(address).toBe("8.8.8.8");
  expect(reply.headers.get("content-type")).toBe("application/octet-stream");
  expect(await consume(reply.chunks())).toBe("bytes");
  reply.close(); reply.close();
  expect(f.counts()).toEqual({ responseCloses: 1, requestCloses: 1, agentCloses: 1, timerClears: 1, starts: 1 });
});
test.each(["http://huggingface.co/x", "https://127.0.0.1/x", "https://evil.test/x", "https://user:secret@huggingface.co/x"])("rejects unsafe URL before client creation: %s", async (value) => {
  const f = fixture();
  await expect(createArtifactHttpTransport(f.runtime)({ url: new URL(value), method: "HEAD" })).rejects.toThrow(/Unapproved/);
  expect(f.counts().starts).toBe(0);
});
test("pre-abort creates no client", async () => {
  const f = fixture(), controller = new AbortController(); controller.abort();
  await expect(createArtifactHttpTransport(f.runtime)({ url: url(), method: "HEAD", signal: controller.signal })).rejects.toThrow();
  expect(f.counts().starts).toBe(0);
});
test("deadline remains active after headers and prevents partial success", async () => {
  const f = fixture(); const reply = await createArtifactHttpTransport(f.runtime)({ url: url(), method: "GET" });
  f.expire();
  await expect(consume(reply.chunks())).rejects.toThrow(/deadline/i);
  expect(f.counts().agentCloses).toBe(1);
});
test("abort after headers destroys owned resources and rejects body consumption", async () => {
  const f = fixture(), controller = new AbortController();
  const reply = await createArtifactHttpTransport(f.runtime)({ url: url(), method: "GET", signal: controller.signal });
  controller.abort();
  await expect(consume(reply.chunks())).rejects.toThrow(/abort/i);
  expect(f.counts().requestCloses).toBe(1);
});
test("request failure after headers is redacted and visible to the stream", async () => {
  const f = fixture(); const reply = await createArtifactHttpTransport(f.runtime)({ url: url(), method: "GET" }); f.error();
  await expect(consume(reply.chunks())).rejects.toThrow("Artifact HTTP request failed");
});
test.each([{ chunks: ["unexpected text"] }, { complete: false }, { failStream: true }])("rejects malformed, truncated or failed binary streams", async (settings) => {
  const f = fixture(settings); const reply = await createArtifactHttpTransport(f.runtime)({ url: url(), method: "GET" });
  const result = await consume(reply.chunks()).catch((error: unknown) => error);
  expect(result).toBeInstanceOf(Error);
  if (result instanceof Error) { expect(result.message).not.toContain("private"); expect(result.cause).toBeUndefined(); }
  expect(f.counts().agentCloses).toBe(1);
});
test("early consumer return closes resources without reading the remaining body", async () => {
  const f = fixture({ chunks: [Buffer.from("a"), Buffer.from("b")] });
  const reply = await createArtifactHttpTransport(f.runtime)({ url: url(), method: "GET" });
  for await (const _chunk of reply.chunks()) break;
  expect(f.counts().responseCloses).toBe(1);
});
test("client startup failures are sanitized and dispose the newly owned agent", async () => {
  const f = fixture();
  const transport = createArtifactHttpTransport({ ...f.runtime, start: () => { throw new Error("fixture-only private startup detail"); } });
  await expect(transport({ url: url(), method: "HEAD" })).rejects.toThrow("Artifact HTTP request failed");
  expect(f.counts().timerClears).toBe(1);
});
test("a request deadline settles a pending header promise", async () => {
  const f = fixture(); const waiting = deferred<void>();
  const start = f.runtime.start;
  if (start === undefined) throw new Error("Missing fixture client");
  const transport = createArtifactHttpTransport({ ...f.runtime, start: (target, options, deliver) => {
    const handle = start(target, options, (response) => { void waiting.promise.then(() => deliver(response)); }); return handle;
  } });
  const result = transport({ url: url(), method: "GET" }).catch((error: unknown) => error);
  f.expire(); waiting.resolve();
  expect(await result).toEqual(new Error("Artifact HTTP deadline exceeded"));
  expect(f.counts().agentCloses).toBe(1);
});
test("the supplied connection lookup cannot admit private DNS answers", async () => {
  const f = fixture();
  const reply = await createArtifactHttpTransport({ ...f.runtime, resolve: async () => [{ address: "127.0.0.1", family: 4 }] })({ url: url(), method: "HEAD" });
  try {
    const lookup = f.options()?.lookup;
    if (lookup === undefined) throw new Error("Missing connection lookup");
    const error = await new Promise((resolve) => lookup("huggingface.co", { all: true }, (failure) => resolve(failure)));
    expect(error).toBeInstanceOf(Error);
  } finally { reply.close(); }
});
test.each([false, true])("sanitizes cleanup failure and preserves a prior stream failure: %s", async (failStream) => {
  const f = fixture({ failStream }); const start = f.runtime.start;
  if (start === undefined) throw new Error("Missing fixture client");
  const transport = createArtifactHttpTransport({ ...f.runtime, start: (target, options, deliver) => start(target, options,
    (response) => deliver({ ...response, destroy: () => { response.destroy(); throw new Error("fixture-only private cleanup detail"); } })) });
  const reply = await transport({ url: url(), method: "GET" });
  await expect(consume(reply.chunks())).rejects.toThrow(failStream ? "Artifact HTTP stream failed" : "Artifact HTTP cleanup failed");
  expect(f.counts().agentCloses).toBe(1);
  expect(f.counts().requestCloses).toBe(1);
});
test("a late header callback cannot leak cleanup errors after the deadline", async () => {
  const f = fixture(); const start = f.runtime.start;
  if (start === undefined) throw new Error("Missing fixture client");
  let deliver: ((response: ArtifactIncoming) => void) | undefined;
  const transport = createArtifactHttpTransport({ ...f.runtime, start: (target, options, callback) => {
    deliver = callback; return start(target, options, () => {});
  } });
  const result = transport({ url: url(), method: "HEAD" }).catch((error: unknown) => error);
  f.expire();
  expect(await result).toEqual(new Error("Artifact HTTP deadline exceeded"));
  const deliverLate = deliver;
  if (deliverLate === undefined) throw new Error("Missing response callback");
  const lateResponse = { ...f.response, destroy: () => { throw new Error("fixture-only private late cleanup detail"); } };
  expect(() => deliverLate(lateResponse)).not.toThrow();
});
test.each([false, true])("scheduler setup failure releases the owned Agent and preserves the sanitized primary error: %s", async (cleanupThrows) => {
  const original = Agent.prototype.destroy;
  const descriptor = Object.getOwnPropertyDescriptor(Agent.prototype, "destroy");
  let destroys = 0, starts = 0;
  Object.defineProperty(Agent.prototype, "destroy", { configurable: true, writable: true, value: function(this: Agent) {
    destroys++; original.call(this);
    if (cleanupThrows) throw new Error("fixture-only private Agent cleanup detail");
  } });
  try {
    const transport = createArtifactHttpTransport({
      schedule: () => { throw new Error("fixture-only private scheduler detail"); },
      start: () => { starts++; throw new Error("Client must not start"); },
    });
    const error = await transport({ url: url(), method: "HEAD" }).catch((failure: unknown) => failure);
    expect(error).toEqual(new Error("Artifact HTTP request failed"));
    if (error instanceof Error) expect(error.cause).toBeUndefined();
    expect(starts).toBe(0);
    expect(destroys).toBe(1);
  } finally {
    if (descriptor) Object.defineProperty(Agent.prototype, "destroy", descriptor);
    else Reflect.deleteProperty(Agent.prototype, "destroy");
  }
});
