import type { ResolvedOnnxConfig } from "./types.js";
import type { OnnxLoadOptions, OnnxRuntimePort } from "./onnx-runtime-types.js";

function assertNever(_value: never): never { throw new Error("Unsupported ONNX provider"); }
export function onnxLoadOptions(
  config: ResolvedOnnxConfig,
  host: Pick<OnnxRuntimePort, "platform" | "arch">,
  device: OnnxLoadOptions["device"] = config.device,
): OnnxLoadOptions {
  const cpuTarget = ["linux", "win32", "darwin"].includes(host.platform) && ["x64", "arm64"].includes(host.arch);
  let targeted: boolean;
  switch (device) {
    case "cpu": targeted = cpuTarget; break;
    case "cuda": targeted = host.platform === "linux" && host.arch === "x64"; break;
    case "dml": targeted = host.platform === "win32" && ["x64", "arm64"].includes(host.arch); break;
    case "coreml": targeted = host.platform === "darwin" && ["x64", "arm64"].includes(host.arch); break;
    default: return assertNever(device);
  }
  if (!targeted) throw new Error("Unsupported ONNX device/platform/architecture target");
  return Object.freeze({ device, providers: Object.freeze([device]), dtype: config.dtype,
    intraOpNumThreads: config.intraOpNumThreads, sequential: true,
    memoryPattern: device !== "dml", localFilesOnly: true });
}
