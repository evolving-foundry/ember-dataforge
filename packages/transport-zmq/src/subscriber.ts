/**
 * ZmqEventSubscriber — TS-side SUB socket of the event channel (doc §8/§39).
 *
 * The Python runtime publishes [topic, envelope] frames on its PUB socket;
 * the topic is the event method name (e.g. "job.progress"), which lets
 * ZeroMQ filter subscriptions before the envelope is even deserialized.
 *
 * P0 may leave eventsEndpoint unconfigured and multiplex events over the
 * command channel instead (doc §9); the subscriber is optional.
 */

import { Subscriber } from "zeromq";
import type { RuntimeMessage } from "@dataforge/protocol";
import { isRuntimeMessage } from "@dataforge/protocol";

export interface EventSubscriberOptions {
  /** e.g. "ipc:///tmp/dataforge-events.sock" or "tcp://127.0.0.1:5556". */
  eventsEndpoint: string;
  /** Topic prefixes to subscribe; empty array means all topics. */
  topics?: string[];
  onProtocolError?: (error: unknown) => void;
}

export class ZmqEventSubscriber {
  private readonly socket: Subscriber;
  private receiveLoop: Promise<void> | null = null;
  private connected = false;
  private closed = false;
  private waiter: (() => void) | null = null;
  private buffered: RuntimeMessage[] = [];

  constructor(private readonly options: EventSubscriberOptions) {
    this.socket = new Subscriber();
  }

  async connect(): Promise<void> {
    if (this.connected) return;
    await this.socket.connect(this.options.eventsEndpoint);
    if (this.options.topics && this.options.topics.length > 0) {
      for (const topic of this.options.topics) {
        this.socket.subscribe(topic);
      }
    } else {
      this.socket.subscribe("");
    }
    this.connected = true;
    this.receiveLoop = this.runReceiveLoop();
  }

  /** Next event, or null when the subscriber has been closed. */
  async next(): Promise<RuntimeMessage | null> {
    while (!this.closed) {
      const message = this.buffered.shift();
      if (message) return message;
      await new Promise<void>((resolve) => {
        this.waiter = resolve;
      });
    }
    return null;
  }

  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    try {
      this.socket.close();
    } finally {
      this.connected = false;
      this.receiveLoop = null;
      this.waiter?.();
    }
  }

  private async runReceiveLoop(): Promise<void> {
    while (!this.closed) {
      let frames: Buffer[];
      try {
        // receive() resolves with one multipart message: Buffer[].
        frames = await this.socket.receive();
      } catch (error) {
        if (this.closed) return;
        this.options.onProtocolError?.(error);
        continue;
      }

      // PUB sends [topic, envelope]; SUB receives both frames.
      const envelopeFrame = frames.length >= 2 ? frames[1] : frames[0];
      let message: unknown;
      try {
        message = JSON.parse(envelopeFrame.toString("utf-8"));
      } catch (error) {
        this.options.onProtocolError?.(error);
        continue;
      }

      if (!isRuntimeMessage(message)) {
        this.options.onProtocolError?.(new Error("event frame is not a RuntimeMessage"));
        continue;
      }

      this.buffered.push(message);
      this.waiter?.();
      this.waiter = null;
    }
  }
}
