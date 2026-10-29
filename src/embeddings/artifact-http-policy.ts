import type { ArtifactSource } from "./artifact-files.js";
import type { OnnxModelSource } from "./types.js";
import { validateArtifactPath } from "./artifact-path.js";

export interface ArtifactHttpReply {
  readonly status: number;
  readonly headers: Headers;
  chunks(): AsyncIterable<Uint8Array>;
  close(): void;
}
export interface ArtifactHttpRequest {
  readonly url: URL;
  readonly method: "HEAD" | "GET";
  readonly signal?: AbortSignal;
}
export type ArtifactHttpTransport = (request: ArtifactHttpRequest) => Promise<ArtifactHttpReply>;
export interface HuggingFaceArtifact {
  readonly source: Extract<OnnxModelSource, { kind: "huggingface" }>;
  readonly path: string;
}

// Reviewed Hub delivery hosts, not a wildcard or a remotely extensible policy.
// https://huggingface.co/docs/hub/models-downloading#downloading-behind-a-proxy-or-firewall
const deliveryHosts = new Set([
  "huggingface.co", "cas-server.xethub.hf.co", "cas-server.xethub-eu.hf.co",
  "transfer.xethub.hf.co", "transfer.xethub-eu.hf.co", "us.aws.cdn.hf.co",
  "us.gcp.cdn.hf.co", "cdn-lfs-us-1.hf.co", "cdn-lfs-eu-1.hf.co",
]);
const hubOrigin = "https://huggingface.co";
function closeReply(response: ArtifactHttpReply, failed: boolean): void {
  try { response.close(); }
  catch { if (!failed) throw new Error("Hugging Face artifact cleanup failed"); }
}
export function validateArtifactDeliveryUrl(url: URL): void {
  if (url.protocol !== "https:" || (url.port !== "" && url.port !== "443")
    || url.username || url.password || url.hash || url.href.length > 16384 || !deliveryHosts.has(url.hostname)) {
    throw new Error("Unapproved HTTPS artifact redirect/host");
  }
}
function digestHeader(headers: Headers): string {
  const raw = headers.get("x-linked-etag") ?? headers.get("etag") ?? "";
  const strong = raw.startsWith("W/") ? raw.slice(2) : raw;
  const value = strong.startsWith('"') && strong.endsWith('"') ? strong.slice(1, -1) : strong;
  if (!/^(?:[a-fA-F0-9]{40}|[a-fA-F0-9]{64})$/.test(value)) throw new Error("Invalid artifact object digest");
  return value.toLowerCase();
}
function metadata(headers: Headers, redirected: boolean) {
  const text = headers.get("x-linked-size") ?? (redirected ? null : headers.get("content-length"));
  if (text === null || !/^\d+$/.test(text) || !Number.isSafeInteger(Number(text))) throw new Error("Invalid or missing artifact size");
  return Object.freeze({ bytes: Number(text), digest: digestHeader(headers) });
}

export async function openHuggingFaceArtifact(
  artifact: HuggingFaceArtifact, transport: ArtifactHttpTransport, signal?: AbortSignal,
): Promise<ArtifactSource | undefined> {
  signal?.throwIfAborted();
  validateArtifactPath(artifact.path);
  if (artifact.path.toLowerCase() === ".qmd-manifest.json") throw new Error("Reserved artifact path");
  const revision = artifact.source.revision.toLowerCase();
  const repo = artifact.source.model.split("/").map(encodeURIComponent).join("/");
  const path = artifact.path.split("/").map(encodeURIComponent).join("/");
  const initial = new URL(`${hubOrigin}/${repo}/resolve/${revision}/${path}`);
  const requireCommit = (headers: Headers, required: boolean): void => {
    const commit = headers.get("x-repo-commit");
    if ((required || commit !== null) && commit?.toLowerCase() !== revision) throw new Error("Artifact commit does not match immutable selection");
  };
  const follow = async (method: "HEAD" | "GET", requestSignal?: AbortSignal): Promise<ArtifactHttpReply | undefined> => {
    let url = initial;
    for (let hops = 0; hops <= 5; hops++) {
      requestSignal?.throwIfAborted();
      validateArtifactDeliveryUrl(url);
      let response: ArtifactHttpReply;
      try { response = await transport({ url, method, signal: requestSignal }); }
      catch { requestSignal?.throwIfAborted(); throw new Error("Hugging Face artifact request failed"); }
      let retained = false;
      let failed = false;
      try {
        requestSignal?.throwIfAborted();
        const encoding = response.headers.get("content-encoding");
        if (encoding !== null && encoding !== "identity") throw new Error("Unsupported artifact content encoding");
        if (url.origin === hubOrigin) requireCommit(response.headers, false);
        if ([301, 302, 303, 307, 308].includes(response.status)) {
          const location = response.headers.get("location");
          if (!location || hops === 5) throw new Error("Invalid or excessive artifact redirects");
          let next: URL;
          try { next = new URL(location, url); }
          catch { throw new Error("Invalid artifact redirect"); }
          validateArtifactDeliveryUrl(next);
          if (url.origin === hubOrigin && next.origin !== hubOrigin) {
            requireCommit(response.headers, true);
            if (method === "HEAD") { retained = true; return response; }
          }
          url = next;
          continue;
        }
        if (method === "HEAD" && url.origin === hubOrigin && response.status === 404) return undefined;
        if (response.status !== 200) throw new Error(`Artifact access failed with status ${response.status}`);
        if (url.origin === hubOrigin) requireCommit(response.headers, true);
        retained = true;
        return response;
      } catch (error) { failed = true; throw error; }
      finally { if (!retained) closeReply(response, failed); }
    }
    throw new Error("Excessive artifact redirects");
  };
  const head = await follow("HEAD", signal);
  if (head === undefined) return undefined;
  let expected: ReturnType<typeof metadata>;
  let failed = false;
  try { expected = metadata(head.headers, head.status !== 200); }
  catch (error) { failed = true; throw error; }
  finally { closeReply(head, failed); }
  return Object.freeze({
    byteLength: expected.bytes,
    ...(expected.digest.length === 64 ? { expectedSha256: expected.digest } : { expectedGitSha1: expected.digest }),
    async *chunks(streamSignal?: AbortSignal) {
      const combined = signal && streamSignal ? AbortSignal.any([signal, streamSignal]) : signal ?? streamSignal;
      const response = await follow("GET", combined);
      if (response === undefined) throw new Error("Artifact disappeared after metadata admission");
      let failed = false;
      try {
        // Final CDN etags need not be Hub object digests; staged bytes are checked against HEAD.
        try {
          for await (const part of response.chunks()) { combined?.throwIfAborted(); yield part; }
          combined?.throwIfAborted();
        } catch { combined?.throwIfAborted(); throw new Error("Hugging Face artifact stream failed"); }
      } catch (error) { failed = true; throw error; }
      finally { closeReply(response, failed); }
    },
  });
}
