interface QueueJob {
  readonly signal?: AbortSignal;
  readonly run: () => Promise<void>;
  readonly cancel: (reason: unknown) => void;
  readonly removeAbort: () => void;
}
export interface OnnxSerialQueue {
  run<T>(operation: () => Promise<T>, signal?: AbortSignal): Promise<T>;
  stop(reason: unknown): void;
}

export function createOnnxSerialQueue(): OnnxSerialQueue {
  const jobs: QueueJob[] = [];
  let draining = false;
  let stopped = false;
  let stopReason: unknown;

  function drain(): void {
    if (draining || stopped) return;
    draining = true;
    void (async () => {
      try {
        while (!stopped) {
          const job = jobs.shift();
          if (job === undefined) break;
          job.removeAbort();
          if (job.signal?.aborted) { job.cancel(job.signal.reason); continue; }
          await job.run();
        }
      } finally {
        draining = false;
        if (!stopped && jobs.length > 0) drain();
      }
    })();
  }

  function run<T>(operation: () => Promise<T>, signal?: AbortSignal): Promise<T> {
    if (stopped) return Promise.reject(stopReason);
    signal?.throwIfAborted();
    return new Promise<T>((resolve, reject) => {
      let job: QueueJob | undefined;
      let settled = false;
      const removeAbort = () => signal?.removeEventListener("abort", abort);
      const resolveOnce = (value: T) => { if (!settled) { settled = true; resolve(value); } };
      const rejectOnce = (reason: unknown) => { if (!settled) { settled = true; reject(reason); } };
      const abort = () => {
        if (job === undefined) return;
        const index = jobs.indexOf(job);
        if (index < 0) return;
        jobs.splice(index, 1);
        job.cancel(signal?.reason);
      };
      job = {
        signal,
        removeAbort,
        cancel: rejectOnce,
        run: async () => {
          try {
            signal?.throwIfAborted();
            resolveOnce(await operation());
          } catch (error) { rejectOnce(error); }
          finally { removeAbort(); }
        },
      };
      signal?.addEventListener("abort", abort, { once: true });
      jobs.push(job);
      if (signal?.aborted) abort();
      if (!settled) drain();
    });
  }

  function stop(reason: unknown): void {
    if (stopped) return;
    stopped = true;
    stopReason = reason;
    for (const job of jobs.splice(0)) { job.removeAbort(); job.cancel(reason); }
  }

  return Object.freeze({ run, stop });
}
