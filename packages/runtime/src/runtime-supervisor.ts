/**
 * RuntimeSupervisor — TS-side owner of the command channel. It binds the
 * ROUTER socket, fans incoming frames into per-runtime transports and the
 * RuntimeRegistry, and enforces the heartbeat timeout (doc §21).
 */

import type { RuntimeMessage } from "@dataforge/protocol";
import { createErrorResponse, createResponse } from "@dataforge/protocol";
import {
  ZmqEventSubscriber,
  ZmqRuntimeRouter,
  ZmqTransport,
} from "@dataforge/transport-zmq";
import { RuntimeRegistry } from "./runtime-registry.js";

export interface SupervisorOptions {
  /** Command channel endpoint the ROUTER binds (doc §25). */
  commandEndpoint: string;
  /** Optional dedicated PUB/SUB event channel (P1, doc §9). */
  eventsEndpoint?: string;
  /** Heartbeat timeout before a runtime is marked unhealthy (§21). */
  heartbeatTimeoutMs?: number;
  /** Interval of the heartbeat watchdog. */
  heartbeatCheckIntervalMs?: number;
  onProtocolError?: (error: unknown, identity: string | null) => void;
  /** Called when a runtime is marked unhealthy by the watchdog. */
  onRuntimeUnhealthy?: (runtimeId: string) => void;
}


/** Envelope payloads are JSON values; treat them as records at the edge. */
function payloadOf(message: RuntimeMessage): Record<string, unknown> {
  return (message.payload ?? {}) as Record<string, unknown>;
}

export class RuntimeSupervisor {
  readonly registry = new RuntimeRegistry();
  /** Command channel endpoint this supervisor's ROUTER is bound to. */
  readonly commandEndpoint: string;
  private readonly router: ZmqRuntimeRouter;
  private readonly transports = new Map<string, ZmqTransport>();
  private subscriber: ZmqEventSubscriber | null = null;
  private watchdog: ReturnType<typeof setInterval> | null = null;
  private started = false;
  private readonly heartbeatTimeoutMs: number;

  constructor(private readonly options: SupervisorOptions) {
    this.heartbeatTimeoutMs = options.heartbeatTimeoutMs ?? 15_000;
    this.commandEndpoint = options.commandEndpoint;
    this.router = new ZmqRuntimeRouter({
      commandEndpoint: options.commandEndpoint,
      onProtocolError: options.onProtocolError
        ? (error, identity) => options.onProtocolError?.(error, identity)
        : undefined,
    });
  }

  async start(): Promise<void> {
    if (this.started) return;
    await this.router.bind();
    this.router.onMessage(({ identity, message }) => {
      void this.handleIncoming(identity, message);
    });

    if (this.options.eventsEndpoint) {
      this.subscriber = new ZmqEventSubscriber({
        eventsEndpoint: this.options.eventsEndpoint,
        onProtocolError: (error) => this.options.onProtocolError?.(error, null),
      });
      await this.subscriber.connect();
      void this.pumpSubscriberEvents();
    }

    const interval = this.options.heartbeatCheckIntervalMs ?? 5_000;
    this.watchdog = setInterval(() => {
      const marked = this.registry.applyHeartbeatTimeout(this.heartbeatTimeoutMs);
      for (const runtimeId of marked) {
        this.options.onRuntimeUnhealthy?.(runtimeId);
      }
    }, interval);
    this.started = true;
  }

  /** Create (or reuse) a transport addressed at one Python runtime. */
  transport(runtimeId: string): ZmqTransport {
    let transport = this.transports.get(runtimeId);
    if (!transport) {
      transport = new ZmqTransport(this.router, runtimeId);
      this.transports.set(runtimeId, transport);
    }
    return transport;
  }

  /** Send a raw envelope to one runtime (lifecycle / admin use). */
  async sendToRuntime(runtimeId: string, message: RuntimeMessage): Promise<void> {
    await this.router.sendToRuntime(runtimeId, message);
  }

  async stop(): Promise<void> {
    if (!this.started) return;
    this.started = false;
    if (this.watchdog) {
      clearInterval(this.watchdog);
      this.watchdog = null;
    }
    for (const transport of this.transports.values()) {
      await transport.close();
    }
    this.transports.clear();
    await this.subscriber?.close();
    this.subscriber = null;
    await this.router.close();
  }

  private async handleIncoming(identity: string, message: RuntimeMessage): Promise<void> {
    switch (message.type) {
      case "event":
        this.applyEvent(message);
        return;
      case "heartbeat": {
        if (message.runtimeId) {
          const hb = payloadOf(message);
          this.registry.heartbeat(
            message.runtimeId,
            typeof hb.activeJobs === "number" ? hb.activeJobs : undefined,
            typeof hb.queueDepth === "number" ? hb.queueDepth : undefined,
          );
        }
        return;
      }
      case "request": {
        if (message.method === "runtime.hello") {
          await this.router.sendToRuntime(
            identity,
            createResponse(message.messageId, { accepted: true }, { method: message.method }),
          );
          const payload = payloadOf(message);
          if (typeof payload.runtimeId === "string") {
            this.registry.register({
              runtimeId: payload.runtimeId,
              capabilities: Array.isArray(payload.capabilities)
                ? (payload.capabilities as string[])
                : [],
              resources: (payload.resources as { cpu: number; memoryGiB: number }) ?? {
                cpu: 0,
                memoryGiB: 0,
              },
              version: typeof payload.version === "string" ? payload.version : undefined,
            });
          }
        } else {
          await this.router.sendToRuntime(
            identity,
            createErrorResponse(
              message.messageId,
              { code: "UNKNOWN_METHOD", message: `supervisor does not handle ${message.method}` },
              { method: message.method },
            ),
          );
        }
        return;
      }
      default:
        // responses and cancels are consumed by ZmqTransport instances.
        return;
    }
  }

  private applyEvent(message: RuntimeMessage): void {
    const payload = payloadOf(message);
    const runtimeId = message.runtimeId ?? (typeof payload.runtimeId === "string" ? payload.runtimeId : undefined);
    if (!runtimeId) return;
    switch (message.method) {
      case "runtime.register":
        this.registry.register({
          runtimeId: String(runtimeId),
          capabilities: Array.isArray(payload.capabilities)
            ? (payload.capabilities as string[])
            : [],
          resources: (payload.resources as { cpu: number; memoryGiB: number }) ?? {
            cpu: 0,
            memoryGiB: 0,
          },
          version: typeof payload.version === "string" ? payload.version : undefined,
        });
        break;
      case "runtime.heartbeat":
        this.registry.heartbeat(
          String(runtimeId),
          typeof payload.activeJobs === "number" ? payload.activeJobs : undefined,
          typeof payload.queueDepth === "number" ? payload.queueDepth : undefined,
        );
        break;
      default:
        break;
    }
  }

  private async pumpSubscriberEvents(): Promise<void> {
    while (this.subscriber) {
      const message = await this.subscriber.next();
      if (!message) return;
      this.applyEvent(message);
    }
  }
}
