import type { ArtifactStage, ArtifactSource } from "./artifact-files.js";
import type { ArtifactFileReader } from "./artifact-local.js";
import type { ArtifactCacheEntry, ArtifactCacheKey, ArtifactCacheSnapshot } from "./artifact-cache.js";
import type { ArtifactHttpTransport, HuggingFaceArtifact } from "./artifact-http-policy.js";
import { join } from "node:path";
import { createArtifactStage } from "./artifact-files.js";
import { createArtifactFileReader } from "./artifact-local.js";
import { createArtifactCache } from "./artifact-cache.js";
import { createArtifactHttpTransport } from "./artifact-http.js";
import { openHuggingFaceArtifact } from "./artifact-http-policy.js";

export interface ArtifactIO {
  stage(): Promise<ArtifactStage>;
  local(directory: string): Promise<ArtifactFileReader>;
  cached(key: ArtifactCacheKey, signal?: AbortSignal): Promise<ArtifactCacheEntry | undefined>;
  download(artifact: HuggingFaceArtifact, signal?: AbortSignal): Promise<ArtifactSource | undefined>;
  publish(key: ArtifactCacheKey, snapshot: ArtifactCacheSnapshot, signal?: AbortSignal): Promise<void>;
}
export function createArtifactIO(root: string, transport: ArtifactHttpTransport = createArtifactHttpTransport()): ArtifactIO {
  const cache = createArtifactCache(join(root, "cache"));
  return Object.freeze({ stage: () => createArtifactStage(join(root, "sessions")), local: createArtifactFileReader,
    cached: cache.find, publish: cache.publish,
    download: (artifact: HuggingFaceArtifact, signal?: AbortSignal) => openHuggingFaceArtifact(artifact, transport, signal),
  });
}
