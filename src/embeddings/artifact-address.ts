import type { LookupAddress } from "node:dns";
import type { LookupFunction } from "node:net";
import { lookup } from "node:dns/promises";
import { BlockList, isIP } from "node:net";

export type ArtifactResolver = (hostname: string) => Promise<readonly LookupAddress[]>;
// Conservative artifact-connection policy, not a general Internet routing classifier.
// IANA IPv4/IPv6 special-purpose registries, reviewed 2026-10-24. Protocol/transition
// ranges are excluded as whole ranges, including some globally reachable exceptions.
const ipv4Blocked = new BlockList();
const ipv4Subnets: readonly (readonly [string, number])[] = [
  ["0.0.0.0", 8], ["10.0.0.0", 8], ["100.64.0.0", 10], ["127.0.0.0", 8],
  ["169.254.0.0", 16], ["172.16.0.0", 12], ["192.0.0.0", 24], ["192.0.2.0", 24],
  ["192.88.99.0", 24], ["192.168.0.0", 16], ["198.18.0.0", 15], ["198.51.100.0", 24],
  ["203.0.113.0", 24], ["224.0.0.0", 3],
];
ipv4Subnets.forEach(([address, prefix]) => ipv4Blocked.addSubnet(address, prefix, "ipv4"));
const ipv6Global = new BlockList();
ipv6Global.addSubnet("2000::", 3, "ipv6");
const ipv6Blocked = new BlockList();
for (const [address, prefix] of [["2001::", 23], ["2001:db8::", 32], ["2002::", 16], ["3fff::", 20]] as const) {
  ipv6Blocked.addSubnet(address, prefix, "ipv6");
}
export function isArtifactAddressAllowed(address: string): boolean {
  if (address.includes("%")) return false;
  switch (isIP(address)) {
    case 4: return !ipv4Blocked.check(address, "ipv4");
    case 6: return ipv6Global.check(address, "ipv6") && !ipv6Blocked.check(address, "ipv6");
    default: return false;
  }
}
export function createArtifactLookup(
  resolver: ArtifactResolver = (hostname) => lookup(hostname, { all: true, verbatim: true }),
): LookupFunction {
  return (hostname, options, callback) => {
    void Promise.resolve().then(() => resolver(hostname)).then((addresses) => {
      if (addresses.length === 0 || addresses.some(({ address, family }) => isIP(address) !== family || !isArtifactAddressAllowed(address))) {
        callback(new Error("Artifact DNS response contains a forbidden address"), "", 0);
        return;
      }
      const family = options.family === "IPv4" ? 4 : options.family === "IPv6" ? 6 : options.family;
      const candidates = family ? addresses.filter((address) => address.family === family) : [...addresses];
      const first = candidates[0];
      if (first === undefined) { callback(new Error("Artifact DNS response has no requested-family address"), "", 0); return; }
      if (options.all) callback(null, candidates);
      else callback(null, first.address, first.family);
    }, () => callback(new Error("Artifact address resolution failed"), "", 0));
  };
}
