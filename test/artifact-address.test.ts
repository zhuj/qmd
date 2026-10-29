import { expect, test } from "vitest";
import type { LookupAddress, LookupOptions } from "node:dns";
import { createArtifactLookup, isArtifactAddressAllowed } from "../src/embeddings/artifact-address.js";
import type { ArtifactResolver } from "../src/embeddings/artifact-address.js";

test.each([
  "0.0.0.0", "0.1.2.3", "10.1.2.3", "100.64.0.1", "100.127.255.255", "127.0.0.1",
  "169.254.169.254", "172.16.0.1", "172.31.255.255", "192.0.0.9", "192.0.2.1", "192.88.99.2",
  "192.168.1.1", "198.18.0.1", "198.19.255.255", "198.51.100.1", "203.0.113.1", "224.0.0.1",
  "239.255.255.255", "240.1.2.3", "255.255.255.255", "::", "::1", "::ffff:127.0.0.1", "::ffff:8.8.8.8",
  "64:ff9b::a00:1", "64:ff9b:1::1", "100::1", "100:0:0:1::1", "2001::1", "2001:2::1",
  "2001:db8::1", "2002:7f00:1::1", "3fff::1", "5f00::1", "fc00::1", "fd12::1", "fe80::1",
  "ff02::1", "fe80::1%eth0", "not-an-address", "127.1", "0177.0.0.1", "0x7f000001", "[::1]",
])("rejects special, ambiguous or unsupported address %s", (address) => {
  expect(isArtifactAddressAllowed(address)).toBe(false);
});
test.each(["8.8.8.8", "1.1.1.1", "100.128.0.1", "172.32.0.1", "198.20.0.1", "2001:4860:4860::8888", "2606:4700:4700::1111"])("admits ordinary unicast policy address %s", (address) => {
  expect(isArtifactAddressAllowed(address)).toBe(true);
});
function lookupResult(options: LookupOptions, resolver: ArtifactResolver) {
  return new Promise<{ error: NodeJS.ErrnoException | null; address: string | LookupAddress[]; family?: number }>((resolve) => {
    createArtifactLookup(resolver)("cdn-lfs-us-1.hf.co", options, (error, address, family) => resolve({ error, address, family }));
  });
}
const publicAnswers = [{ address: "8.8.8.8", family: 4 }, { address: "2606:4700:4700::1111", family: 6 }];
test("returns an address array for Node family-autoselection lookup", async () => {
  const hosts: string[] = [];
  const result = await lookupResult({ all: true }, async (hostname) => { hosts.push(hostname); return publicAnswers; });
  expect(result.error).toBeNull();
  expect(result.address).toEqual(publicAnswers);
  expect(hosts).toEqual(["cdn-lfs-us-1.hf.co"]);
});
test("returns one address and family for ordinary lookup", async () => {
  const result = await lookupResult({}, async () => publicAnswers);
  expect(result.error).toBeNull();
  expect(result.address).toBe("8.8.8.8");
  expect(result.family).toBe(4);
});
test.each([6, "IPv6"] as const)("honors requested family %s", async (family) => {
  const result = await lookupResult({ all: true, family }, async () => publicAnswers);
  expect(result.error).toBeNull();
  expect(result.address).toEqual([{ address: "2606:4700:4700::1111", family: 6 }]);
});
test.each([
  { answers: [] }, { answers: [{ address: "127.0.0.1", family: 4 }] },
  { answers: [...publicAnswers, { address: "10.0.0.1", family: 4 }] },
  { answers: [{ address: "::ffff:169.254.169.254", family: 6 }] },
  { answers: [{ address: "8.8.8.8", family: 6 }] },
])("rejects the whole unsafe resolver answer without exposing a candidate", async ({ answers }) => {
  const result = await lookupResult({ all: true }, async () => answers);
  expect(result.error?.message).toMatch(/address|resolution/i);
  expect(result.address).toBe("");
});
test("fails when no requested-family candidate is available", async () => {
  const result = await lookupResult({ family: 6 }, async () => [{ address: "8.8.8.8", family: 4 }]);
  expect(result.error?.message).toMatch(/family/i);
  expect(result.address).toBe("");
});
test("redacts resolver failures instead of attaching an unsafe cause", async () => {
  const result = await lookupResult({}, async () => { throw new Error("private internal resolver detail"); });
  expect(result.error?.message).toBe("Artifact address resolution failed");
  expect(result.error?.cause).toBeUndefined();
});
test("redacts synchronous resolver failure through the lookup callback", async () => {
  const result = await lookupResult({}, () => { throw new Error("private synchronous resolver detail"); });
  expect(result.error?.message).toBe("Artifact address resolution failed");
  expect(result.error?.cause).toBeUndefined();
});
