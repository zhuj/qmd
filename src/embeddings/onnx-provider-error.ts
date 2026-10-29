import type { OnnxLoadOptions } from "./onnx-runtime-types.js";

/** These prefixes are emitted before model loading by the pinned Transformers/ORT JS layers. */
export function isOnnxProviderInitializationError(
  error: unknown,
  requestedDevice: OnnxLoadOptions["device"],
): boolean {
  if (requestedDevice === "cpu") return false;
  const provider = requestedDevice;
  try {
    if (!(error instanceof Error)) return false;
    // Transformers.js 4.3.1 and ORT common 1.30.0 emit these before model loading.
    return error.message.startsWith(`no available backend found. ERR: [${provider}] `)
      || error.message.startsWith(`Unsupported device: "${provider}". Should be one of: `);
  } catch {
    return false;
  }
}
