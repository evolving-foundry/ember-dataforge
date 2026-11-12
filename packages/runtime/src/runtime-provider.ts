/**
 * Runtime Contract (doc §5) — the language-neutral seam between the TS
 * product core and any execution runtime.
 *
 * This interface knows nothing about ZMQ, Ray, or Python. Implementations
 * combine a RuntimeProvider with a RuntimeTransport (doc §6).
 */

import type {
  DatasetMetadata,
  DatasetRef,
  ExecutionContext,
  ExecutionResult,
  OperationSpec,
  RuntimeMessage,
} from "@dataforge/protocol";

export interface ExecutionRuntime {
  start(): Promise<void>;

  execute(
    operation: OperationSpec,
    input?: DatasetRef,
    context?: ExecutionContext,
  ): Promise<ExecutionResult>;

  cancel(jobId: string): Promise<void>;

  inspectDataset(ref: DatasetRef): Promise<DatasetMetadata>;

  releaseDataset(ref: DatasetRef): Promise<void>;

  /** Health check; implementers should not throw on unhealthy runtimes. */
  health(): Promise<{ runtimeId: string; healthy: boolean; details?: Record<string, unknown> }>;

  events(): AsyncIterable<RuntimeMessage>;

  dispose(): Promise<void>;
}

/**
 * RuntimeProvider (doc §28) — creates/owns an ExecutionRuntime instance.
 * e.g. PythonRuntimeProvider spawns a long-lived Python process; a remote
 * provider would connect over HTTP/gRPC instead.
 */
export interface RuntimeProvider {
  /** Stable identity used for routing and registry bookkeeping. */
  readonly runtimeId: string;
  start(): Promise<ExecutionRuntime>;
  dispose(): Promise<void>;
}
