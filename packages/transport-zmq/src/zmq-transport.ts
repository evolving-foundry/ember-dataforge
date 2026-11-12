/**
 * ZmqTransport — RuntimeTransport implementation over the ROUTER/DEALER
 * command channel (doc §6/§7). One ZmqTransport talks to one Python
 * runtime identified by its ZeroMQ routing identity.
 */

import type { RuntimeMessage } from "@dataforge/protocol";
import { createCancel, createRequest } from "@dataforge/protocol";
import { ZmqRuntimeRouter } from "./router.js";
import { TransportError, type RequestOptions, type RuntimeTransport, type TransportHealth } from "./transport.js";

interface PendingRequest {
  resolve: (payload: RuntimeMessage) => void;
  reject: (error: TransportError) => void;
  timer: ReturnType<typeof setTimeout>;
}

/** Simple async queue used to fan events() out to consumers. */
class EventQueue {
  private items: RuntimeMessage[] = [];
  private waiters: Array<{ resolve: () => void }> = [];
  private closed = false;

  push(message: RuntimeMessage): void {
    if (this.closed) return;
    this.items.push(message);
    this.flushWaiters();
  }

  close(): void {
    this.closed = true;
    this.flushWaiters();
  }

  async next(): Promise<IteratorResult<RuntimeMessage>> {
    for (;;) {
      const item = this.items.shift();
      if (item) return { value: item, done: false };
      if (this.closed) return { value: undefined, done: true };
      await new Promise<void>((resolve) => this.waiters.push({ resolve }));
    }
  }

  private flushWaiters(): void {
    while (this.waiters.length > 0) {
      this.waiters.shift()?.resolve();
    }
  }
}

export interface ZmqTransportOptions {
  /** Timeout applied when a request does not set timeoutMs. */
  defaultTimeoutMs?: number;
}

export class ZmqTransport implements RuntimeTransport {
  private readonly pending = new Map<string, PendingRequest>();
  private readonly eventQueues = new Set<EventQueue>();
  private readonly unsubscribe: () => void;
  private readonly defaultTimeoutMs: number;
  private connected = false;

  constructor(
    private readonly router: ZmqRuntimeRouter,
    private readonly runtimeId: string,
    options: ZmqTransportOptions = {},
  ) {
    this.defaultTimeoutMs = options.defaultTimeoutMs ?? 120_000;
    this.unsubscribe = router.onMessage(({ identity, message }) => {
      if (identity !== this.runtimeId) return;
      this.handleIncoming(message);
    });
  }

  async connect(): Promise<void> {
    // The ROUTER is bound by the supervisor; the peer may connect shortly
    // after. Connectivity failures surface as request timeouts.
    this.connected = true;
  }

  /**
   * Wait until the peer's first frame establishes the ROUTER→DEALER route.
   * ZeroMQ silently drops messages sent to an identity that is not yet in
   * the routing table, so callers must await this before the first request
   * unless the peer is known to have spoken already (e.g. runtime.hello).
   */
  async waitForPeer(timeoutMs = 30_000): Promise<void> {
    const deadline = Date.now() + timeoutMs;
    while (this.router.lastSeenMs(this.runtimeId) === undefined) {
      if (Date.now() > deadline) {
        throw new TransportError(
          "PEER_UNREACHABLE",
          `runtime ${this.runtimeId} sent no frames within ${timeoutMs}ms`,
          { runtimeId: this.runtimeId },
        );
      }
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
  }

  async request<TRequest, TResponse>(
    method: string,
    payload: TRequest,
    options?: RequestOptions,
  ): Promise<TResponse> {
    const timeoutMs = options?.timeoutMs ?? this.defaultTimeoutMs;
    const requestMessage = createRequest(method, payload, {
      runtimeId: this.runtimeId,
      jobId: options?.jobId,
    });

    const response = await this.sendAndWait(requestMessage, timeoutMs);
    if (response.error) {
      throw new TransportError(
        response.error.code,
        response.error.message,
        response.error.details,
      );
    }
    return response.payload as TResponse;
  }

  async cancel(requestId: string): Promise<void> {
    const cancelMessage = createCancel(requestId, { runtimeId: this.runtimeId });
    await this.router.sendToRuntime(this.runtimeId, cancelMessage);
  }

  events(): AsyncIterable<RuntimeMessage> {
    const queue = new EventQueue();
    this.eventQueues.add(queue);
    const queues = this.eventQueues;
    return {
      [Symbol.asyncIterator]() {
        return {
          next: () => queue.next(),
          return: async () => {
            queues.delete(queue);
            return { value: undefined, done: true };
          },
        };
      },
    };
  }

  async health(): Promise<TransportHealth> {
    const lastSeen = this.router.lastSeenMs(this.runtimeId);
    const lastFrameAgeMs = lastSeen === undefined ? undefined : Date.now() - lastSeen;
    return {
      connected: this.connected && lastSeen !== undefined,
      lastFrameAgeMs,
      details: { runtimeId: this.runtimeId },
    };
  }

  async close(): Promise<void> {
    this.unsubscribe();
    for (const [, pending] of this.pending) {
      clearTimeout(pending.timer);
      pending.reject(new TransportError("TRANSPORT_CLOSED", "transport closed"));
    }
    this.pending.clear();
    for (const queue of this.eventQueues) {
      queue.close();
    }
    this.eventQueues.clear();
    this.connected = false;
  }

  private sendAndWait(
    message: RuntimeMessage,
    timeoutMs: number,
  ): Promise<RuntimeMessage> {
    return new Promise<RuntimeMessage>((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(message.messageId);
        reject(
          new TransportError("REQUEST_TIMEOUT", `no response for ${message.method} within ${timeoutMs}ms`, {
            messageId: message.messageId,
            method: message.method,
          }),
        );
      }, timeoutMs);

      this.pending.set(message.messageId, { resolve, reject, timer });
      void this.router.sendToRuntime(this.runtimeId, message).catch((error: unknown) => {
        clearTimeout(timer);
        this.pending.delete(message.messageId);
        reject(
          error instanceof TransportError
            ? error
            : new TransportError("SEND_FAILED", String(error)),
        );
      });
    });
  }

  private handleIncoming(message: RuntimeMessage): void {
    if (message.type === "response" && message.correlationId) {
      const pending = this.pending.get(message.correlationId);
      if (pending) {
        this.pending.delete(message.correlationId);
        clearTimeout(pending.timer);
        pending.resolve(message);
        return;
      }
      // Late or duplicated response — ignore.
      return;
    }

    // Events and heartbeats fan out to all subscribers.
    for (const queue of this.eventQueues) {
      queue.push(message);
    }
  }
}
