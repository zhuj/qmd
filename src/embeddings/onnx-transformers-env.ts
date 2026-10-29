import { z } from "zod";

/** Reject process-global Transformers settings that bypass the per-store artifact boundary. */
const transformersEnvironment = z.object({
  allowRemoteModels: z.boolean(),
  allowLocalModels: z.boolean(),
  useFS: z.boolean(),
  useFSCache: z.boolean(),
  useBrowserCache: z.boolean(),
  useCustomCache: z.boolean(),
  experimental_useCrossOriginStorage: z.boolean(),
}).passthrough();

export function assertTransformersEnvironment(value: unknown): void {
  const parsed = transformersEnvironment.safeParse(value);
  if (!parsed.success) throw new Error("Unsupported Transformers.js environment configuration");
  const env = parsed.data;
  if (!env.allowLocalModels || !env.useFS || env.useBrowserCache || env.useCustomCache || env.experimental_useCrossOriginStorage) {
    throw new Error("Transformers.js must use local filesystem loading without shared caches");
  }
}
