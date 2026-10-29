/** Portable relative artifact names; never normalize an ambiguous native path. */
export function validateArtifactPath(path: string): void {
  if (!path || path.length > 4096 || !/^[A-Za-z0-9_.-]+(?:\/[A-Za-z0-9_.-]+)*$/.test(path)
    || path.split("/").some((part) => part.endsWith(".") || /^(con|prn|aux|nul|com[1-9]|lpt[1-9])(?:\.|$)/i.test(part))) {
    throw new Error("Unsafe artifact location/path");
  }
}
