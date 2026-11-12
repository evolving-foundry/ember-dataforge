import { describe, expect, it } from "vitest";
import {
  createEvent,
  createRequest,
  createResponse,
  createCancel,
  createHeartbeat,
  createErrorResponse,
  createRuntimeError,
  isRuntimeMessage,
  RUNTIME_PROTOCOL,
  RUNTIME_PROTOCOL_VERSION,
} from "@dataforge/protocol";

describe("protocol envelope", () => {
  it("creates requests with frozen protocol identity", () => {
    const msg = createRequest("operation.execute", { operation: { name: "x" } });
    expect(msg.protocol).toBe(RUNTIME_PROTOCOL);
    expect(msg.version).toBe(RUNTIME_PROTOCOL_VERSION);
    expect(msg.type).toBe("request");
    expect(msg.method).toBe("operation.execute");
    expect(msg.messageId).toMatch(/^msg-/);
    expect(typeof msg.timestamp).toBe("number");
  });

  it("correlates responses to requests", () => {
    const req = createRequest("runtime.health", {});
    const res = createResponse(req.messageId, { ok: true }, { method: req.method });
    expect(res.correlationId).toBe(req.messageId);
    expect(res.payload).toEqual({ ok: true });
  });

  it("builds error responses", () => {
    const req = createRequest("runtime.health", {});
    const res = createErrorResponse(
      req.messageId,
      createRuntimeError("BOOM", "failed"),
      { method: req.method },
    );
    expect(res.correlationId).toBe(req.messageId);
    expect(res.error).toEqual({ code: "BOOM", message: "failed" });
  });

  it("creates events, cancels and heartbeats", () => {
    const ev = createEvent("job.progress", { jobId: "job-1" }, { jobId: "job-1" });
    expect(ev.type).toBe("event");
    expect(ev.method).toBe("job.progress");

    const cancel = createCancel(ev.messageId, { jobId: "job-1" });
    expect(cancel.type).toBe("cancel");
    expect(cancel.correlationId).toBe(ev.messageId);

    const hb = createHeartbeat({ activeJobs: 1 }, { runtimeId: "rt-1" });
    expect(hb.type).toBe("heartbeat");
    expect(hb.method).toBe("runtime.heartbeat");
  });

  it("validates incoming envelopes structurally", () => {
    expect(isRuntimeMessage({ protocol: "dataforge-runtime", messageId: "m", type: "request" })).toBe(true);
    expect(isRuntimeMessage({ protocol: "other", messageId: "m", type: "request" })).toBe(false);
    expect(isRuntimeMessage(null)).toBe(false);
    expect(isRuntimeMessage("hello")).toBe(false);
  });
});
