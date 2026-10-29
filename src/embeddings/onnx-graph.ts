import { validateArtifactPath } from "./artifact-path.js";

export interface OnnxGraphFile {
  readonly byteLength: number;
  read(offset: number, length: number): Promise<Uint8Array>;
}

export interface OnnxExternalData {
  readonly location: string;
  readonly offset: number;
  readonly length?: number;
  readonly checksum?: string;
}

type Container = "model" | "graph" | "node" | "attribute" | "tensor" | "sparse";
const children: Readonly<Record<Container, ReadonlyMap<number, Container>>> = {
  model: new Map([[7, "graph"]]),
  graph: new Map([[1, "node"], [5, "tensor"], [15, "sparse"]]),
  node: new Map([[5, "attribute"]]),
  attribute: new Map([[5, "tensor"], [6, "graph"], [10, "tensor"], [11, "graph"], [22, "sparse"], [23, "sparse"]]),
  tensor: new Map(), sparse: new Map([[1, "tensor"], [2, "tensor"]]),
};
const singularChildren: Readonly<Record<Container, readonly number[]>> = {
  model: [7], graph: [], node: [], attribute: [5, 6, 22], tensor: [], sparse: [1, 2],
};
// Field layout: https://github.com/onnx/onnx/blob/v1.19.1/onnx/onnx.proto
// Unknown containers fail closed; training/functions/device contexts are not encoder snapshots.
const fields: Readonly<Record<Container, readonly number[]>> = {
  model: [1, 2, 3, 4, 5, 6, 7, 8, 14], graph: [1, 2, 5, 10, 11, 12, 13, 14, 15, 16],
  node: [1, 2, 3, 4, 5, 6, 7, 8, 9], attribute: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 20, 21, 22, 23],
  tensor: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16], sparse: [1, 2, 3],
};

function externalReference(metadata: ReadonlyMap<string, string>): OnnxExternalData {
  const location = metadata.get("location");
  if (location === undefined) throw new Error("Missing ONNX external data location/path");
  validateArtifactPath(location);
  const integer = (key: string): number | undefined => {
    const text = metadata.get(key);
    if (text === undefined) return undefined;
    if (!/^\d+$/.test(text) || !Number.isSafeInteger(Number(text))) throw new Error("Invalid ONNX external metadata integer");
    return Number(text);
  };
  const checksum = metadata.get("checksum");
  if (checksum !== undefined && !/^[a-fA-F0-9]{40}$/.test(checksum)) throw new Error("Invalid ONNX external checksum metadata");
  const length = integer("length");
  return Object.freeze({ location, offset: integer("offset") ?? 0,
    ...(length === undefined ? {} : { length }), ...(checksum === undefined ? {} : { checksum: checksum.toLowerCase() }) });
}

export async function inspectOnnxExternalData(
  file: OnnxGraphFile, signal?: AbortSignal,
): Promise<readonly OnnxExternalData[]> {
  signal?.throwIfAborted();
  if (!Number.isSafeInteger(file.byteLength) || file.byteLength <= 0) throw new Error("Invalid ONNX graph byte length");
  let position = 0;
  let cache = new Uint8Array();
  let cacheView = new DataView(cache.buffer);
  let cacheStart = 0;
  let fieldCount = 0;
  let metadataBytes = 0;
  let graphCount = 0;
  let irVersion: bigint | undefined;
  const references: OnnxExternalData[] = [];
  const decoder = new TextDecoder("utf-8", { fatal: true });

  async function byte(end: number): Promise<number> {
    signal?.throwIfAborted();
    if (position >= end) throw new Error("Truncated ONNX protobuf");
    if (position < cacheStart || position >= cacheStart + cache.length) {
      cacheStart = position;
      const size = Math.min(8192, file.byteLength - position);
      cache = new Uint8Array(await file.read(position, size));
      if (cache.length !== size) throw new Error("Truncated ONNX graph read");
      cacheView = new DataView(cache.buffer, cache.byteOffset, cache.byteLength);
    }
    return cacheView.getUint8(position++ - cacheStart);
  }

  async function unsigned(end: number): Promise<bigint> {
    let value = 0n;
    for (let index = 0; index < 10; index++) {
      const part = await byte(end);
      if (index === 9 && part > 1) throw new Error("Invalid ONNX protobuf varint");
      value |= BigInt(part & 127) << BigInt(index * 7);
      if (part < 128) return value;
    }
    throw new Error("Invalid ONNX protobuf varint");
  }

  async function header(end: number): Promise<{ id: number; wire: number }> {
    if (++fieldCount > 1_000_000) throw new Error("ONNX graph field limit exceeded");
    const tag = await unsigned(end);
    if (tag < 8n || tag > 0xffffffffn) throw new Error("Invalid ONNX protobuf field tag");
    return { id: Math.floor(Number(tag) / 8), wire: Number(tag % 8n) };
  }

  async function payloadEnd(end: number): Promise<number> {
    const length = await unsigned(end);
    if (length > BigInt(end - position)) throw new Error("Truncated ONNX protobuf payload");
    return position + Number(length);
  }

  async function string(end: number): Promise<string> {
    const length = end - position;
    metadataBytes += length;
    if (length > 4096 || metadataBytes > 16 * 1024 * 1024) throw new Error("ONNX metadata string limit exceeded");
    const value = await file.read(position, length);
    if (value.length !== length) throw new Error("Truncated ONNX metadata");
    position = end;
    return decoder.decode(value);
  }

  async function externalEntry(end: number): Promise<readonly [string, string]> {
    const values = new Map<number, string>();
    while (position < end) {
      const { id, wire } = await header(end);
      if ((id !== 1 && id !== 2) || wire !== 2 || values.has(id)) throw new Error("Invalid ONNX external metadata entry");
      values.set(id, await string(await payloadEnd(end)));
    }
    const key = values.get(1);
    const value = values.get(2);
    if (key === undefined || value === undefined || !["location", "offset", "length", "checksum"].includes(key)) {
      throw new Error("Unsupported ONNX external metadata key");
    }
    return [key, value];
  }

  async function walk(frame: Readonly<{ kind: Container; end: number; depth: number }>): Promise<void> {
    const { kind, end, depth } = frame;
    if (depth > 32) throw new Error("ONNX graph nesting depth limit exceeded");
    const metadata = new Map<string, string>();
    const seenChildren = new Set<number>();
    let dataLocation = 0n;
    let inline = false;
    let domain = "";
    while (position < end) {
      const { id, wire } = await header(end);
      if (!fields[kind].includes(id)) throw new Error(`Unsupported ONNX ${kind} field ${id}`);
      const child = children[kind].get(id);
      if (child !== undefined && wire !== 2) throw new Error("Invalid ONNX protobuf container");
      const scalarField = (kind === "model" && id === 1) || (kind === "tensor" && id === 14);
      const messageField = (kind === "tensor" && id === 13) || (kind === "node" && id === 7);
      if ((scalarField && wire !== 0) || (messageField && wire !== 2)) throw new Error("Invalid ONNX closure field wire encoding");
      if (singularChildren[kind].includes(id)) {
        if (seenChildren.has(id)) throw new Error("Duplicate singular ONNX graph container");
        seenChildren.add(id);
      }
      if (kind === "tensor" && [4, 5, 6, 7, 9, 10, 11].includes(id)) inline = true;
      switch (wire) {
        case 0: {
          const value = await unsigned(end);
          if (kind === "model" && id === 1) {
            if (irVersion !== undefined || value < 1n || value > 12n) throw new Error("Unsupported ONNX IR version");
            irVersion = value;
          }
          if (kind === "tensor" && id === 14) {
            if (value > 1n) throw new Error("Unsupported ONNX external data location");
            dataLocation = value;
          }
          break;
        }
        case 1: position += 8; break;
        case 5: position += 4; break;
        case 2: {
          const next = await payloadEnd(end);
          if (child !== undefined) {
            if (kind === "model" && ++graphCount !== 1) throw new Error("ONNX model must contain exactly one graph");
            await walk({ kind: child, end: next, depth: depth + 1 });
          } else if (kind === "tensor" && id === 13) {
            const [key, value] = await externalEntry(next);
            if (metadata.has(key)) throw new Error("Duplicate ONNX external metadata key");
            metadata.set(key, value);
          } else if (kind === "node" && id === 7) {
            domain = await string(next);
          } else {
            position = next;
          }
          break;
        }
        default: throw new Error("Unsupported ONNX protobuf wire type");
      }
      if (position > end) throw new Error("Truncated ONNX protobuf field");
    }
    if (kind === "node" && domain !== "" && domain !== "ai.onnx") throw new Error("Unsupported ONNX operator domain");
    if (kind === "tensor" && (metadata.size > 0 || dataLocation === 1n)) {
      if (dataLocation !== 1n || inline) throw new Error("Inconsistent ONNX external/inline tensor metadata");
      if (references.length >= 65536) throw new Error("ONNX external reference limit exceeded");
      references.push(externalReference(metadata));
    }
  }

  await walk({ kind: "model", end: file.byteLength, depth: 0 });
  signal?.throwIfAborted();
  if (irVersion === undefined || graphCount !== 1) throw new Error("ONNX model requires an IR version and graph");
  return Object.freeze(references);
}
