export interface Deferred<T> {
  readonly promise: Promise<T>;
  resolve(value: T): void;
  reject(reason: Error): void;
}

export function deferred<T>(): Deferred<T> {
  return Promise.withResolvers<T>();
}
