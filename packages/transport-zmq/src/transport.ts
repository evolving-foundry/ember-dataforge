/**
 * RuntimeTransport — the seam between Runtime Contract and any transport
 * (doc §6). ZmqTransport is the default local implementation; HTTP/gRPC
 * transports can substitute it without touching the contract.
 */

import type { RuntimeMessage } from "@dataforge/protocol";

export interface RequestOptions {
  /** Fail the request if no response arrives within this window. */
  timeoutMs?: number;
  runtimeId?: string;
  jobId?: string;
}

export interface TransportHealth {
  connected: boolean;
  /** ms since last frame from the peer, when known. */
  lastFrameAgeMs?: number;
  details?: Record<string, unknown>;
}

export interface RuntimeTransport {
  connect(): Promise<void>;

  /**
   * Send a request and await its correlated response payload.
   * Throws TransportError when the response carries an error envelope
   * or the request times out.
   */
  request<TRequest, TResponse>(
    method: string,
    payload: TRequest,
    options?: RequestOptions,
  ): Promise<TResponse>;

  /** Best-effort cancellation of an in-flight request. */
  cancel(requestId: string): Promise<void>;

  /** Stream of unsolicited runtime messages (events, heartbeats). */
  events(): AsyncIterable<RuntimeMessage>;

  health(): Promise<TransportHealth>;

  close(): Promise<void>;
}

export class TransportError extends Error {
  readonly code: string;
  readonly details?: Record<string, unknown>;

  constructor(code: string, message: string, details?: Record<string, unknown>) {
    super(message);
    this.name = "TransportError";
    this.code = code;
    this.details = details;
  }
}
