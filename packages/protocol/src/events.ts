/**
 * Runtime Protocol — runtime event catalog (Phase 0 freeze).
 *
 * Event method names and their payloads (doc §8/§15/§19/§21). Events flow
 * Python → TS either multiplexed over the DEALER/ROUTER command channel
 * (P0) or on the PUB/SUB event channel (P1); the envelope is identical.
 */

import type { JobStatus, RuntimeCapability } from "./operation.js";
import type { DatasetRef } from "./dataset.js";

export interface RuntimeRegisteredPayload {
  runtimeId: string;
  capabilities: RuntimeCapability[];
  resources: {
    cpu: number;
    memoryGiB: number;
    npu?: number;
    gpu?: number;
  };
  version?: string;
}

export interface RuntimeHeartbeatPayload {
  runtimeId: string;
  activeJobs: number;
  cpu?: number;
  memory?: number;
  npu?: number;
  gpu?: number;
  rayStatus?: string;
  queueDepth?: number;
}

export interface RuntimeWarningPayload {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface RuntimeMetricsPayload {
  runtimeId: string;
  metrics: Record<string, number | string>;
}

export interface JobEventPayload {
  jobId: string;
  operation?: string;
  status?: JobStatus;
  /** 0..1 progress ratio or absolute counts, operation-defined. */
  progress?: { current?: number; total?: number; ratio?: number };
  error?: { code: string; message: string; details?: Record<string, unknown> };
  result?: { datasetId?: string } & Record<string, unknown>;
  metrics?: Record<string, unknown>;
}

export interface DatasetEventPayload {
  dataset: DatasetRef;
  reason?: string;
}

export interface ArtifactEventPayload {
  artifactId: string;
  kind: string;
  location?: string;
  jobId?: string;
}

export interface ModelEventPayload {
  modelId: string;
  name?: string;
}

export type RuntimeEventMethod =
  | "runtime.ready"
  | "runtime.register"
  | "runtime.heartbeat"
  | "runtime.warning"
  | "runtime.metrics"
  | "job.started"
  | "job.progress"
  | "job.completed"
  | "job.failed"
  | "job.cancelled"
  | "dataset.created"
  | "dataset.released"
  | "artifact.created"
  | "model.loaded";

export type RuntimeEventPayloadMap = {
  "runtime.ready": Record<string, never> | RuntimeRegisteredPayload;
  "runtime.register": RuntimeRegisteredPayload;
  "runtime.heartbeat": RuntimeHeartbeatPayload;
  "runtime.warning": RuntimeWarningPayload;
  "runtime.metrics": RuntimeMetricsPayload;
  "job.started": JobEventPayload;
  "job.progress": JobEventPayload;
  "job.completed": JobEventPayload;
  "job.failed": JobEventPayload;
  "job.cancelled": JobEventPayload;
  "dataset.created": DatasetEventPayload;
  "dataset.released": DatasetEventPayload;
  "artifact.created": ArtifactEventPayload;
  "model.loaded": ModelEventPayload;
};

/** Command-channel method names (doc §8/§34). */
export type RuntimeCommandMethod =
  | "runtime.hello"
  | "runtime.health"
  | "operation.execute"
  | "job.cancel"
  | "dataset.inspect"
  | "dataset.release"
  | "pipeline.execute";
