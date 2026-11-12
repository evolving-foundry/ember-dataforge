/**
 * ZmqRuntimeRouter — the TS-side ROUTER socket of the command channel
 * (doc §7/§8). Frames:
 *
 *   ROUTER side (here):   [identity, envelope, ...optional frames]
 *   DEALER side (python): [envelope, ...optional frames]
 *
 * The envelope is a JSON-serialized RuntimeMessage (P0 wire format §23).
 */

import { Router } from "zeromq";
import type { RuntimeMessage } from "@dataforge/protocol";
import { isRuntimeMessage } from "@dataforge/protocol";

export interface RouterOptions {
  /** e.g. "ipc:///tmp/dataforge-runtime.sock" or "tcp://127.0.0.1:5555". */
  commandEndpoint: string;
  /** Log protocol violations instead of throwing into the receive loop. */
  onProtocolError?: (error: unknown, identity: string | null) => void;
}

export interface RouterFrame {
  /** ZeroMQ routing identity of the DEALER peer. */
  identity: string;
  message: RuntimeMessage;
  /** Extra frames after the envelope (reserved, doc §24). */
  extraFrames: Buffer[];
}

type MessageHandler = (frame: RouterFrame) => void;

export class ZmqRuntimeRouter {
  private readonly socket: Router;
  private readonly options: RouterOptions;
  private readonly handlers = new Set<MessageHandler>();
  private readonly lastSeen = new Map<string, number>();
  /** Raw identity bytes per string form (random ZMQ identities are not UTF-8). */
  private readonly identityBytes = new Map<string, Buffer>();
  private receiveLoop: Promise<void> | null = null;
  private bound = false;
  private closed = false;

  constructor(options: RouterOptions) {
    this.options = options;
    this.socket = new Router();
  }

  async bind(): Promise<void> {
    if (this.bound) return;
    await this.socket.bind(this.options.commandEndpoint);
    this.bound = true;
    this.receiveLoop = this.runReceiveLoop();
  }

  /** Register a handler for every incoming DEALER frame. */
  onMessage(handler: MessageHandler): () => void {
    this.handlers.add(handler);
    return () => this.handlers.delete(handler);
  }

  /** Send an envelope to one DEALER peer by routing identity. */
  async sendToRuntime(identity: string, message: RuntimeMessage): Promise<void> {
    if (!this.bound) {
      throw new Error("router not bound");
    }
    const envelope = Buffer.from(JSON.stringify(message), "utf-8");
    const identityFrame =
      this.identityBytes.get(identity) ?? Buffer.from(identity, "utf-8");
    await this.socket.send([identityFrame, envelope]);
  }

  /** Identities of all DEALER peers seen so far. */
  knownIdentities(): string[] {
    return [...this.lastSeen.keys()];
  }

  lastSeenMs(identity: string): number | undefined {
    return this.lastSeen.get(identity);
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    try {
      this.socket.close();
    } finally {
      this.handlers.clear();
    }
    this.receiveLoop = null;
  }

  private async runReceiveLoop(): Promise<void> {
    while (!this.closed) {
      let frames: Buffer[];
      try {
        // receive() resolves with one multipart message: Buffer[].
        frames = await this.socket.receive();
      } catch (error) {
        if (this.closed) return;
        this.options.onProtocolError?.(error, null);
        continue;
      }

      const [identityFrame, envelopeFrame, ...rest] = frames;
      if (!identityFrame || !envelopeFrame) {
        this.options.onProtocolError?.(
          new Error(`expected [identity, envelope] frames, got ${frames.length}`),
          null,
        );
        continue;
      }

      const identity = identityFrame.toString("utf-8");
      this.identityBytes.set(identity, Buffer.from(identityFrame));
      this.lastSeen.set(identity, Date.now());

      let message: unknown;
      try {
        message = JSON.parse(envelopeFrame.toString("utf-8"));
      } catch (error) {
        this.options.onProtocolError?.(error, identity);
        continue;
      }

      if (!isRuntimeMessage(message)) {
        this.options.onProtocolError?.(new Error("frame is not a RuntimeMessage"), identity);
        continue;
      }

      const frame: RouterFrame = {
        identity,
        message,
        extraFrames: rest,
      };
      for (const handler of this.handlers) {
        try {
          handler(frame);
        } catch (error) {
          this.options.onProtocolError?.(error, identity);
        }
      }
    }
  }
}
