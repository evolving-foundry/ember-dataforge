/**
 * Cross-language end-to-end: TS RuntimeSupervisor (ROUTER) + real
 * `python -m runtime.zmq_server` (DEALER) over ipc, exercising the doc §34
 * P0 method set through the PythonRuntimeProvider.
 */

import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { RuntimeSupervisor } from "@dataforge/runtime";
import { PythonRuntimeProvider } from "@dataforge/runtime-python";
import type { ExecutionRuntime } from "@dataforge/runtime";

const RUNTIME_ID = "python-e2e";
const ENDPOINT = `ipc:///tmp/df-e2e-${process.pid}.sock`;

let supervisor: RuntimeSupervisor;
let provider: PythonRuntimeProvider;
let runtime: ExecutionRuntime;

beforeAll(async () => {
  supervisor = new RuntimeSupervisor({
    commandEndpoint: ENDPOINT,
    heartbeatTimeoutMs: 60_000,
    heartbeatCheckIntervalMs: 30_000,
  });
  await supervisor.start();

  provider = new PythonRuntimeProvider({
    supervisor,
    runtimeId: RUNTIME_ID,
    pythonPath: ".venv/bin/python",
    projectRoot: process.cwd(),
    capabilities: ["ray-data", "test"],
    resources: { cpu: 2, memoryGiB: 4 },
  });
  runtime = await provider.start();
}, 90_000);

afterAll(async () => {
  await provider.dispose();
  await supervisor.stop();
}, 60_000);

describe("supervisor ↔ python runtime e2e", () => {
  it("registers the runtime via the runtime.hello handshake", () => {
    const entry = supervisor.registry.get(RUNTIME_ID);
    expect(entry).toBeDefined();
    expect(entry?.status).toBe("registered");
    expect(entry?.capabilities).toContain("ray-data");
  });

  it("answers runtime.health through the full protocol stack", async () => {
    const health = await runtime.health();
    expect(health.healthy).toBe(true);
    expect(health.runtimeId).toBe(RUNTIME_ID);
  });

  it("rejects unknown operations with a runtime error", async () => {
    await expect(
      runtime.execute({ name: "no_such_op_xyz", params: {} }),
    ).rejects.toMatchObject({
      code: "RUNTIME_ERROR",
      message: expect.stringContaining("Unknown operation"),
    });
  });

  it("rejects dataset.inspect for unknown datasets", async () => {
    await expect(
      runtime.inspectDataset({ id: "ds-missing-xyz", backend: "ray" }),
    ).rejects.toMatchObject({ code: "RUNTIME_ERROR" });
  });

  it("reports dataset.release for unknown datasets as not released", async () => {
    const payload = await supervisor
      .transport(RUNTIME_ID)
      .request<unknown, { datasetId: string; released: boolean }>(
        "dataset.release",
        { datasetId: "ds-missing-xyz" },
      );
    expect(payload.released).toBe(false);
  });
});
