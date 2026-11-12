/**
 * Runtime Protocol — unified message envelope (Phase 0 freeze).
 *
 * The protocol belongs to DataForge, not to ZMQ and not to Python.
 * Every frame crossing the runtime fabric is one RuntimeMessage serialized
 * as JSON / UTF-8 (P0 wire format, see doc §23).
 */

export const RUNTIME_PROTOCOL = "dataforge-runtime";
export const RUNTIME_PROTOCOL_VERSION = "1";

export type RuntimeMessageType =
  | "request"
  | "response"
  | "event"
  | "cancel"
  | "heartbeat";

export interface RuntimeError {
  code: string;
  message: string;
  /** Optional machine-readable details (e.g. failing method, stack hint). */
  details?: Record<string, unknown>;
}

/**
 * Unified message envelope (doc §10).
 */
export interface RuntimeMessage<T = unknown> {
  protocol: typeof RUNTIME_PROTOCOL;
  version: typeof RUNTIME_PROTOCOL_VERSION;

  messageId: string;

  type: RuntimeMessageType;

  /** e.g. "operation.execute", "runtime.health", "job.progress" */
  method?: string;

  /** For responses: messageId of the request this answers. */
  correlationId?: string;

  runtimeId?: string;
  jobId?: string;

  /** Unix epoch milliseconds. */
  timestamp: number;

  payload?: T;

  error?: RuntimeError;
}

let messageCounter = 0;

export function newMessageId(): string {
  messageCounter += 1;
  const uuid = (globalThis as { crypto?: { randomUUID?: () => string } })
    .crypto?.randomUUID?.();
  const unique = uuid ?? `id-${Date.now().toString(36)}-${process.pid}-${messageCounter}`;
  return `msg-${unique}`;
}

export function nowMs(): number {
  return Date.now();
}

function base<T>(
  type: RuntimeMessageType,
  extra: Partial<RuntimeMessage<T>>,
): RuntimeMessage<T> {
  return {
    protocol: RUNTIME_PROTOCOL,
    version: RUNTIME_PROTOCOL_VERSION,
    messageId: newMessageId(),
    type,
    timestamp: nowMs(),
    ...extra,
  };
}

export function createRequest<M extends string, P>(
  method: M,
  payload: P,
  opts: { runtimeId?: string; jobId?: string; messageId?: string } = {},
): RuntimeMessage<P> {
  const msg = base<P>("request", { method, payload, runtimeId: opts.runtimeId, jobId: opts.jobId });
  if (opts.messageId) msg.messageId = opts.messageId;
  return msg;
}

export function createResponse<P>(
  correlationId: string,
  payload: P,
  opts: { runtimeId?: string; jobId?: string; method?: string } = {},
): RuntimeMessage<P> {
  return base<P>("response", { correlationId, payload, runtimeId: opts.runtimeId, jobId: opts.jobId, method: opts.method });
}

export function createErrorResponse(
  correlationId: string,
  error: RuntimeError,
  opts: { runtimeId?: string; jobId?: string; method?: string } = {},
): RuntimeMessage<never> {
  return base<never>("response", { correlationId, error, runtimeId: opts.runtimeId, jobId: opts.jobId, method: opts.method });
}

export function createEvent<M extends string, P>(
  method: M,
  payload: P,
  opts: { runtimeId?: string; jobId?: string } = {},
): RuntimeMessage<P> {
  return base<P>("event", { method, payload, runtimeId: opts.runtimeId, jobId: opts.jobId });
}

export function createCancel(
  targetMessageId: string,
  opts: { runtimeId?: string; jobId?: string } = {},
): RuntimeMessage<never> {
  return base<never>("cancel", {
    correlationId: targetMessageId,
    runtimeId: opts.runtimeId,
    jobId: opts.jobId,
  });
}

export function createHeartbeat<P>(
  payload: P,
  opts: { runtimeId?: string } = {},
): RuntimeMessage<P> {
  return base<P>("heartbeat", { method: "runtime.heartbeat", payload, runtimeId: opts.runtimeId });
}

export function createRuntimeError(
  code: string,
  message: string,
  details?: Record<string, unknown>,
): RuntimeError {
  return { code, message, details };
}

/** Structural validation for incoming frames (defensive, at the boundary). */
export function isRuntimeMessage(value: unknown): value is RuntimeMessage {
  if (typeof value !== "object" || value === null) return false;
  const v = value as Record<string, unknown>;
  return (
    v.protocol === RUNTIME_PROTOCOL &&
    typeof v.messageId === "string" &&
    typeof v.type === "string" &&
    (v.timestamp === undefined || typeof v.timestamp === "number")
  );
}
