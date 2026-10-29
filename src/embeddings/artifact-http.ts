import type { RequestOptions } from "node:https";
import { Agent, request as httpsRequest } from "node:https";
import type { ArtifactResolver } from "./artifact-address.js";
import { createArtifactLookup } from "./artifact-address.js";
import type { ArtifactHttpTransport } from "./artifact-http-policy.js";
import { validateArtifactDeliveryUrl } from "./artifact-http-policy.js";

export interface ArtifactIncoming extends AsyncIterable<unknown> {
  readonly statusCode?: number;
  readonly rawHeaders: readonly string[];
  readonly complete: boolean;
  on(event: "error", listener: (error: Error) => void): void;
  destroy(): void;
}
export interface ArtifactRequestHandle {
  on(event: "error", listener: (error: Error) => void): void;
  end(): void;
  destroy(): void;
}
// Narrow platform ports let ordinary tests exercise ownership without sockets.
export interface ArtifactHttpRuntime {
  readonly start?: (url: URL, options: RequestOptions, response: (incoming: ArtifactIncoming) => void) => ArtifactRequestHandle;
  readonly schedule?: (callback: () => void, milliseconds: number) => () => void;
  readonly resolve?: ArtifactResolver;
}
export function createArtifactHttpTransport(runtime: ArtifactHttpRuntime = {}): ArtifactHttpTransport {
  const start = runtime.start ?? httpsRequest;
  const lookup = createArtifactLookup(runtime.resolve);
  const schedule = runtime.schedule ?? ((callback, milliseconds) => {
    const timer = setTimeout(callback, milliseconds);
    timer.unref();
    return () => clearTimeout(timer);
  });
  return async (input) => {
    const url = new URL(input.url.href);
    validateArtifactDeliveryUrl(url);
    input.signal?.throwIfAborted();
    // Never inherit a global agent's socket pool, proxy configuration or TLS sessions.
    const agent = new Agent({ keepAlive: false, maxSockets: 1, maxTotalSockets: 1, maxCachedSessions: 0, lookup });
    return new Promise((resolve, reject) => {
      let handle: ArtifactRequestHandle | undefined, incoming: ArtifactIncoming | undefined;
      let closed = false, delivered = false, consumed = false;
      let failure: Error | undefined, cleanupFailure: Error | undefined;
      let cancelDeadline = () => {};
      function release(): Error | undefined {
        if (closed) return cleanupFailure;
        closed = true;
        // Every owned release is attempted even if another resource's cleanup fails.
        for (const dispose of [cancelDeadline, () => input.signal?.removeEventListener("abort", abort),
          () => incoming?.destroy(), () => handle?.destroy(), () => agent.destroy()]) {
          try { dispose(); } catch { cleanupFailure ??= new Error("Artifact HTTP cleanup failed"); }
        }
        return cleanupFailure;
      }
      function close(): void { const error = release(); if (error) throw error; }
      function fail(message: string): void {
        if (closed) return;
        failure = new Error(message);
        release();
        if (!delivered) reject(failure);
      }
      function abort(): void { fail("Artifact HTTP request aborted"); }
      try {
        cancelDeadline = schedule(() => fail("Artifact HTTP deadline exceeded"), 300_000);
        input.signal?.addEventListener("abort", abort, { once: true });
        if (input.signal?.aborted) { abort(); return; }
        handle = start(url, {
          method: input.method, agent, lookup, servername: url.hostname, rejectUnauthorized: true,
          maxHeaderSize: 16384, headers: { "Accept-Encoding": "identity" },
        }, (response) => {
          response.on("error", () => fail("Artifact HTTP response failed"));
          if (closed) {
            try { response.destroy(); } catch { cleanupFailure ??= new Error("Artifact HTTP cleanup failed"); }
            return;
          }
          incoming = response;
          try {
            const status = response.statusCode;
            if (status === undefined || !Number.isInteger(status) || status < 100 || status > 599) throw new Error("Invalid HTTP status");
            const headers = new Headers();
            for (let index = 0; index < response.rawHeaders.length; index += 2) {
              const name = response.rawHeaders[index], value = response.rawHeaders[index + 1];
              if (name === undefined || value === undefined) throw new Error("Invalid HTTP headers");
              headers.append(name, value);
            }
            delivered = true;
            resolve(Object.freeze({ status, headers, close,
              async *chunks() {
                if (failure) throw failure;
                if (closed || consumed) throw new Error("Artifact HTTP reply is closed or consumed");
                consumed = true;
                try {
                  for await (const chunk of response) {
                    if (failure) throw failure;
                    if (!(chunk instanceof Uint8Array)) throw new Error("Invalid binary response");
                    yield chunk;
                  }
                  if (failure) throw failure;
                  if (!response.complete) throw new Error("Incomplete HTTP response");
                } catch {
                  failure ??= new Error("Artifact HTTP stream failed");
                  throw failure;
                } finally {
                  const cleanup = release();
                  if (cleanup && !failure) throw cleanup;
                }
              },
            }));
          } catch { fail("Artifact HTTP response failed"); }
        });
        handle.on("error", () => fail("Artifact HTTP request failed"));
        handle.end();
      } catch { fail("Artifact HTTP request failed"); }
    });
  };
}
