import { expect, test } from "vitest";
import { inspectOnnxExternalData, type OnnxGraphFile } from "../src/embeddings/onnx-graph.js";

function varint(value: number | bigint): Uint8Array {
  let remaining = BigInt(value);
  const bytes: number[] = [];
  do {
    const part = Number(remaining & 127n);
    remaining >>= 7n;
    bytes.push(part | (remaining ? 128 : 0));
  } while (remaining);
  return Uint8Array.from(bytes);
}
function bytes(...values: readonly Uint8Array[]): Uint8Array { return Buffer.concat(values); }
function field(id: number, value: Uint8Array | string): Uint8Array {
  const payload = typeof value === "string" ? new TextEncoder().encode(value) : value;
  return bytes(varint(id * 8 + 2), varint(payload.length), payload);
}
function scalar(id: number, value: number | bigint): Uint8Array { return bytes(varint(id * 8), varint(value)); }
function entry(key: string, value: string): Uint8Array { return field(13, bytes(field(1, key), field(2, value))); }
function tensor(location: string, extra: Uint8Array = new Uint8Array()): Uint8Array {
  return bytes(scalar(14, 1), entry("location", location), extra);
}
function model(graph: Uint8Array): Uint8Array { return bytes(scalar(1, 10), field(7, graph)); }
function file(value: Uint8Array): OnnxGraphFile {
  return { byteLength: value.length, read: async (offset, length) => value.slice(offset, offset + length) };
}

test("finds top-level external tensors with exact offset/length metadata", async () => {
  const graph = field(5, tensor("weights.bin", bytes(entry("offset", "4"), entry("length", "8"))));
  expect(await inspectOnnxExternalData(file(model(graph)))).toEqual([{ location: "weights.bin", offset: 4, length: 8 }]);
});

test("finds tensors nested inside graph attributes and sparse tensors", async () => {
  const attributes = bytes(
    field(5, tensor("constant.bin")),
    field(6, field(5, tensor("nested/data.bin"))),
    field(10, tensor("repeated.bin")),
    field(11, field(5, tensor("repeated-graph.bin"))),
    field(22, bytes(field(1, tensor("sparse-values.bin")), field(2, tensor("sparse-indices.bin")))),
    field(23, field(1, tensor("repeated-sparse.bin"))),
  );
  const graph = bytes(field(1, field(5, attributes)), field(15, field(1, tensor("initializer-sparse.bin"))));
  expect((await inspectOnnxExternalData(file(model(graph)))).map((reference) => reference.location)).toEqual([
    "constant.bin", "nested/data.bin", "repeated.bin", "repeated-graph.bin", "sparse-values.bin",
    "sparse-indices.bin", "repeated-sparse.bin", "initializer-sparse.bin",
  ]);
});

test("skips a large inline weight payload with bounded small reads", async () => {
  const start = bytes(scalar(1, 10), varint(58), varint(1_000_010), varint(42), varint(1_000_006), varint(74), varint(1_000_002));
  let largestRead = 0;
  const source: OnnxGraphFile = {
    byteLength: start.length + 1_000_002,
    read: async (offset, length) => {
      largestRead = Math.max(largestRead, length);
      return Uint8Array.from({ length }, (_, index) => start[offset + index] ?? 0);
    },
  };
  expect(await inspectOnnxExternalData(source)).toEqual([]);
  expect(largestRead).toBeLessThanOrEqual(8192);
});

test.each(["../escape.bin", "/absolute.bin", "a/../escape.bin", "a\\escape.bin", "C:escape.bin", "a//b", "./data", "https://host/x", "bad\u0000name"])(
  "rejects unsafe or ambiguous external location %s", async (location) => {
    await expect(inspectOnnxExternalData(file(model(field(5, tensor(location)))))).rejects.toThrow(/location|path/i);
  },
);

test.each([
  bytes(scalar(14, 1)),
  bytes(entry("location", "weights.bin")),
  tensor("weights.bin", entry("location", "other.bin")),
  tensor("weights.bin", entry("basepath", "/private")),
  tensor("weights.bin", entry("offset", "-1")),
  tensor("weights.bin", entry("length", "9007199254740992")),
  tensor("weights.bin", entry("checksum", "not-a-sha1")),
  tensor("weights.bin", field(9, new Uint8Array([1]))),
])("rejects inconsistent or unsupported external metadata %#", async (value) => {
  await expect(inspectOnnxExternalData(file(model(field(5, value))))).rejects.toThrow(/external|metadata|inline/i);
});

test("preserves valid checksum metadata for staged byte verification", async () => {
  const value = tensor("weights.bin", entry("checksum", "a".repeat(40)));
  expect(await inspectOnnxExternalData(file(model(field(5, value))))).toEqual([
    { location: "weights.bin", offset: 0, checksum: "a".repeat(40) },
  ]);
});

test.each([
  bytes(scalar(1, 99), field(7, new Uint8Array())),
  bytes(scalar(1, 10), field(20, new Uint8Array()), field(7, new Uint8Array())),
  bytes(scalar(1, 10), field(25, new Uint8Array()), field(7, new Uint8Array())),
  bytes(scalar(1, 10), field(26, new Uint8Array()), field(7, new Uint8Array())),
  bytes(model(new Uint8Array()), field(27, new Uint8Array())),
  model(field(17, new Uint8Array())),
  model(field(1, bytes(field(4, "EPContext"), field(7, "com.microsoft")))),
  bytes(model(new Uint8Array()), field(7, new Uint8Array())),
])("rejects unsupported model layouts instead of guessing closure %#", async (value) => {
  await expect(inspectOnnxExternalData(file(value))).rejects.toThrow(/unsupported|graph|version/i);
});

test.each([
  new Uint8Array(), new Uint8Array([0]), new Uint8Array([8, 128]), new Uint8Array([11]),
  new Uint8Array([8, ...Array.from({ length: 11 }, () => 255)]),
  bytes(scalar(1, 10), new Uint8Array([58, 20, 1])),
])("rejects truncated/malformed protobuf %#", async (value) => {
  await expect(inspectOnnxExternalData(file(value))).rejects.toThrow(/ONNX|protobuf|graph|version/i);
});

test("bounds recursive graph depth", async () => {
  let graph = field(5, tensor("weights.bin"));
  for (let index = 0; index < 40; index++) graph = field(1, field(5, field(6, graph)));
  await expect(inspectOnnxExternalData(file(model(graph)))).rejects.toThrow(/depth|nest/i);
});

test("bounds metadata strings before allocating them", async () => {
  const source = model(field(5, tensor("a".repeat(4097))));
  await expect(inspectOnnxExternalData(file(source))).rejects.toThrow(/string limit/i);
});

test("cancels an inspection after an admitted asynchronous read", async () => {
  const controller = new AbortController();
  const source = model(field(5, tensor("weights.bin")));
  let reads = 0;
  const inspecting = inspectOnnxExternalData({
    byteLength: source.length,
    read: async (offset, length) => {
      reads++;
      controller.abort(new Error("cancelled during read"));
      return source.slice(offset, offset + length);
    },
  }, controller.signal);
  await expect(inspecting).rejects.toThrow("cancelled during read");
  expect(reads).toBe(1);
});

test("aborted inspections stop before any file read", async () => {
  const controller = new AbortController();
  controller.abort(new Error("inspection cancelled"));
  let reads = 0;
  await expect(inspectOnnxExternalData({ byteLength: 1, read: async () => { reads++; return new Uint8Array(); } }, controller.signal))
    .rejects.toThrow("inspection cancelled");
  expect(reads).toBe(0);
});

test.each([
  model(field(5, scalar(13, 1))),
  model(field(5, field(14, new Uint8Array([1])))),
  model(field(1, scalar(7, 1))),
  bytes(model(new Uint8Array()), field(1, new Uint8Array([1]))),
])("rejects wrong wire types on closure-relevant fields %#", async (value) => {
  await expect(inspectOnnxExternalData(file(value))).rejects.toThrow(/wire|encoding/i);
});

test.each([
  model(field(15, bytes(field(1, tensor("weights.bin")), field(1, field(9, new Uint8Array([1])))))),
  model(field(1, field(5, bytes(field(5, tensor("weights.bin")), field(5, field(9, new Uint8Array([1]))))))),
])("rejects split singular-message tensor state %#", async (value) => {
  await expect(inspectOnnxExternalData(file(value))).rejects.toThrow(/duplicate|singular/i);
});

test("retains genuinely repeated tensor fields", async () => {
  const attribute = bytes(field(10, tensor("first.bin")), field(10, tensor("second.bin")));
  expect(await inspectOnnxExternalData(file(model(field(1, field(5, attribute)))))).toEqual([
    { location: "first.bin", offset: 0 }, { location: "second.bin", offset: 0 },
  ]);
});

test.each(["weights.bin.", "nested./weights.bin", "NUL.bin", "con", "COM1.bin", "aux/data.bin"])(
  "rejects portable filename ambiguity %s", async (location) => {
    await expect(inspectOnnxExternalData(file(model(field(5, tensor(location)))))).rejects.toThrow(/location|path/i);
  },
);

test.each(["weights.v1.bin", "nested/weights-0.bin"])("accepts ordinary portable location %s", async (location) => {
  expect(await inspectOnnxExternalData(file(model(field(5, tensor(location)))))).toEqual([{ location, offset: 0 }]);
});
