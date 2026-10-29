import { expect, test } from "vitest";
import { openHuggingFaceArtifact } from "../src/embeddings/artifact-http-policy.js";
import type { ArtifactHttpReply, ArtifactHttpRequest, ArtifactHttpTransport } from "../src/embeddings/artifact-http-policy.js";

const revision = "a".repeat(40);
const digest = "b".repeat(64);
const artifact = { source: { kind: "huggingface", model: "fixture/encoder", revision }, path: "onnx/model.onnx" } as const;
function reply(status = 200, headers: Record<string, string> = {}, body = "bytes") {
  let closes = 0;
  const response: ArtifactHttpReply = {
    status, headers: new Headers(headers),
    async *chunks() { yield Buffer.from(body); }, close() { closes++; },
  };
  return { response, closed: () => closes };
}
const metadata = { "x-repo-commit": revision, etag: `"${digest}"`, "content-length": "5" };
function fixture(responses: readonly ArtifactHttpReply[]) {
  const requests: ArtifactHttpRequest[] = [];
  const transport: ArtifactHttpTransport = async (request) => {
    requests.push(request);
    const response = responses[requests.length - 1];
    if (!response) throw new Error("Unexpected fixture request");
    return response;
  };
  return { requests, transport };
}
async function consume(source: Awaited<ReturnType<typeof openHuggingFaceArtifact>>): Promise<string> {
  if (!source) throw new Error("Missing fixture source");
  const parts: Uint8Array[] = [];
  for await (const part of source.chunks()) parts.push(part);
  return Buffer.concat(parts).toString();
}

test("pins metadata and defers streamed GET until consumption", async () => {
  const head = reply(200, metadata), get = reply(200, metadata);
  const io = fixture([head.response, get.response]);
  const source = await openHuggingFaceArtifact(artifact, io.transport);
  expect(source?.byteLength).toBe(5);
  expect(source?.expectedSha256).toBe(digest);
  expect(io.requests.map((request) => request.method)).toEqual(["HEAD"]);
  expect(io.requests[0]?.url.pathname).toBe(`/fixture/encoder/resolve/${revision}/onnx/model.onnx`);
  expect(await consume(source)).toBe("bytes");
  expect(io.requests.map((request) => request.method)).toEqual(["HEAD", "GET"]);
  expect([head.closed(), get.closed()]).toEqual([1, 1]);
});
test("uses Git blob SHA1 separately from raw SHA1", async () => {
  const head = reply(200, { ...metadata, etag: `W/"${"c".repeat(40)}"` });
  const source = await openHuggingFaceArtifact(artifact, fixture([head.response]).transport);
  expect(source?.expectedGitSha1).toBe("c".repeat(40));
  expect(source?.expectedSha256).toBeUndefined();
});
test("accepts approved CDN redirects without persisting signed locations", async () => {
  const location = "https://cdn-lfs-us-1.hf.co/object?signature=secret";
  const head = reply(302, { "x-repo-commit": revision, "x-linked-etag": digest, "x-linked-size": "5", location });
  const redirect = reply(302, { "x-repo-commit": revision, location });
  const final = reply(200, { "content-length": "5" });
  const io = fixture([head.response, redirect.response, final.response]);
  const source = await openHuggingFaceArtifact(artifact, io.transport);
  expect(await consume(source)).toBe("bytes");
  expect(io.requests.map((request) => request.url.hostname)).toEqual(["huggingface.co", "huggingface.co", "cdn-lfs-us-1.hf.co"]);
  expect(JSON.stringify(source)).not.toContain("secret");
  expect([head.closed(), redirect.closed(), final.closed()]).toEqual([1, 1, 1]);
});
test("follows Hub routing redirects before receiving pinned metadata", async () => {
  const routing = reply(307, { location: `/api/resolve-cache/models/fixture/encoder/${revision}/config.json` });
  const io = fixture([routing.response, reply(200, metadata).response]);
  expect((await openHuggingFaceArtifact(artifact, io.transport))?.byteLength).toBe(5);
  expect(io.requests).toHaveLength(2);
  expect(routing.closed()).toBe(1);
});
test.each(["../escape", "/absolute", "onnx\\model.onnx", "NUL.bin"])("rejects unsafe declared path %s before transport", async (path) => {
  const io = fixture([]);
  await expect(openHuggingFaceArtifact({ ...artifact, path }, io.transport)).rejects.toThrow(/path|location/i);
  expect(io.requests).toHaveLength(0);
});
test.each([
  { ...metadata, "x-repo-commit": "c".repeat(40) }, { etag: digest, "content-length": "5" },
  { ...metadata, etag: "not-a-digest" }, { ...metadata, "content-length": "-1" },
  { ...metadata, "content-length": "9007199254740992" }, { ...metadata, "content-encoding": "gzip" },
])("rejects unbound, malformed or encoded metadata", async (headers) => {
  const head = reply(200, headers);
  await expect(openHuggingFaceArtifact(artifact, fixture([head.response]).transport)).rejects.toThrow(/commit|digest|size|encoding/i);
  expect(head.closed()).toBe(1);
});
test.each([
  "http://cdn-lfs-us-1.hf.co/file", "https://evil.example/file", "https://huggingface.co.evil.example/file",
  "https://user:password@huggingface.co/file", "https://huggingface.co:444/file", "https://127.0.0.1/file",
  "https://cdn-lfs-us-1.hf.co/file#fragment",
])("rejects unapproved redirect without contacting it", async (location) => {
  const head = reply(302, { ...metadata, location });
  const io = fixture([head.response]);
  await expect(openHuggingFaceArtifact(artifact, io.transport)).rejects.toThrow(/redirect|host|HTTPS/i);
  expect(io.requests).toHaveLength(1);
  expect(head.closed()).toBe(1);
});
test("rejects leaving the Hub without immutable metadata", async () => {
  const head = reply(302, { location: "https://cdn-lfs-us-1.hf.co/file" });
  await expect(openHuggingFaceArtifact(artifact, fixture([head.response]).transport)).rejects.toThrow(/commit|metadata/i);
});
test("bounds routing redirect loops", async () => {
  const response = reply(302, { location: "/loop" });
  const io = fixture(Array.from({ length: 10 }, () => response.response));
  await expect(openHuggingFaceArtifact(artifact, io.transport)).rejects.toThrow(/redirect/i);
  expect(io.requests.length).toBeLessThanOrEqual(6);
  expect(response.closed()).toBe(io.requests.length);
});
test("reports an origin 404 as missing but rejects authorization failure", async () => {
  expect(await openHuggingFaceArtifact(artifact, fixture([reply(404).response]).transport)).toBeUndefined();
  await expect(openHuggingFaceArtifact(artifact, fixture([reply(403).response]).transport)).rejects.toThrow(/status|access/i);
});
test("rejects a changed producer or compressed GET before yielding bytes", async () => {
  const variants: readonly Record<string, string>[] = [{ ...metadata, "x-repo-commit": "c".repeat(40) }, { "content-encoding": "gzip" }];
  for (const headers of variants) {
    const get = reply(200, headers);
    const io = fixture([reply(200, metadata).response, get.response]);
    const source = await openHuggingFaceArtifact(artifact, io.transport);
    await expect(consume(source)).rejects.toThrow(/commit|encoding/i);
    expect(get.closed()).toBe(1);
  }
});
test("aborts before metadata and binds streaming cancellation to transport", async () => {
  const canceled = new AbortController(); canceled.abort(new Error("stop"));
  const io = fixture([]);
  await expect(openHuggingFaceArtifact(artifact, io.transport, canceled.signal)).rejects.toThrow("stop");
  expect(io.requests).toHaveLength(0);
  const stream = new AbortController();
  const get = reply(200, metadata);
  const second = fixture([reply(200, metadata).response, get.response]);
  const source = await openHuggingFaceArtifact(artifact, second.transport);
  if (!source) throw new Error("Missing source");
  for await (const part of source.chunks(stream.signal)) { expect(part.byteLength).toBe(5); break; }
  expect(second.requests[1]?.signal).toBe(stream.signal);
  expect(get.closed()).toBe(1);
});
test("does not expose signed URLs from a transport error", async () => {
  const transport: ArtifactHttpTransport = async () => { throw new Error("secret?signature=credential"); };
  await expect(openHuggingFaceArtifact(artifact, transport)).rejects.toThrow(/^Hugging Face artifact request failed$/);
});

function throwingCleanup(response: ArtifactHttpReply) {
  let attempts = 0;
  return {
    response: { ...response, close() { attempts++; throw new Error("https://cdn-lfs-us-1.hf.co/object?signature=fixture-only"); } },
    attempts: () => attempts,
  };
}
test("sanitizes successful HEAD cleanup failures", async () => {
  const head = throwingCleanup(reply(200, metadata).response);
  await expect(openHuggingFaceArtifact(artifact, fixture([head.response]).transport)).rejects.toThrow(/^Hugging Face artifact cleanup failed$/);
  expect(head.attempts()).toBe(1);
});
test("sanitizes successful GET cleanup failures", async () => {
  const get = throwingCleanup(reply(200, metadata).response);
  const source = await openHuggingFaceArtifact(artifact, fixture([reply(200, metadata).response, get.response]).transport);
  await expect(consume(source)).rejects.toThrow(/^Hugging Face artifact cleanup failed$/);
  expect(get.attempts()).toBe(1);
});
test("preserves redirect rejection when cleanup also fails", async () => {
  const head = throwingCleanup(reply(302, { ...metadata, location: "https://evil.example/file" }).response);
  const io = fixture([head.response]);
  await expect(openHuggingFaceArtifact(artifact, io.transport)).rejects.toThrow(/^Unapproved HTTPS artifact redirect\/host$/);
  expect(head.attempts()).toBe(1);
  expect(io.requests).toHaveLength(1);
});
test("preserves sanitized stream failure when cleanup also fails", async () => {
  const get = throwingCleanup({ ...reply(200, metadata).response, async *chunks() { throw new Error("secret-stream-signature"); } });
  const source = await openHuggingFaceArtifact(artifact, fixture([reply(200, metadata).response, get.response]).transport);
  await expect(consume(source)).rejects.toThrow(/^Hugging Face artifact stream failed$/);
  expect(get.attempts()).toBe(1);
});
