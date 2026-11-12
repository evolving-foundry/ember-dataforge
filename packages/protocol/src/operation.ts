/**
 * Runtime Protocol — operation, job and result references (Phase 0 freeze).
 *
 * These are the ONLY operation-level types allowed to cross the runtime
 * boundary. Ray Datasets, tensors, weights never appear here (doc §12/§13).
 */

export interface OperationSpec {
  /** Registered operation name, e.g. "image_captioning". */
  name: string;
  /** Operation version, e.g. "1". */
  version?: string;
  /** Operation parameters (JSON-serializable only). */
  params?: Record<string, unknown>;
}

export interface ResourceRequirements {
  cpu?: number;
  memoryGiB?: number;
  npu?: number;
  gpu?: number;
}

/** Capabilities a runtime advertises / an operation requires (doc §16). */
export type RuntimeCapability =
  | "ray-data"
  | "torch"
  | "torch-npu"
  | "transformers"
  | "opencv"
  | "librosa"
  | string;

export type JobStatus =
  | "queued"
  | "running"
  | "completed"
  | "failed"
  | "cancelled";

export interface JobRef {
  jobId: string;
  runtimeId: string;
  status: JobStatus;
  /** Operation name for observability. */
  operation?: string;
  createdAt?: number;
}

export interface ExecutionContext {
  /** Job id supplied by the scheduler (generated if omitted). */
  jobId?: string;
  /** Wait for the final result instead of returning a JobRef early. */
  awaitResult?: boolean;
  /** Deadline for awaiting the result, in milliseconds. */
  timeoutMs?: number;
  resources?: ResourceRequirements;
  /** Capabilities the target runtime must have (P0: informational). */
  requiredCapabilities?: RuntimeCapability[];
  /** Free-form tracking/experiment context (JSON-serializable). */
  metadata?: Record<string, unknown>;
}

export interface ExecutionResult {
  jobId: string;
  /** Reference to the output dataset, when the operation produced one. */
  output?: { datasetId?: string } & Record<string, unknown>;
  /** Resulting job status; "completed" for awaitResult calls. */
  status?: JobStatus;
  /** Operation-level metrics (rows processed, duration, ...). */
  metrics?: Record<string, unknown>;
}

export interface ArtifactRef {
  artifactId: string;
  runtimeId: string;
  kind: string;
  location?: string;
  metadata?: Record<string, unknown>;
}

export interface ModelRef {
  modelId: string;
  runtimeId: string;
  /** Model name as understood by the Python runtime, e.g. "Qwen2.5-VL". */
  name?: string;
  metadata?: Record<string, unknown>;
}

export interface RuntimeHealth {
  runtimeId: string;
  healthy: boolean;
  activeJobs: number;
  /** Milliseconds since last heartbeat from the runtime. */
  lastHeartbeatAgeMs: number;
  details?: Record<string, unknown>;
}
