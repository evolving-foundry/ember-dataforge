import { afterEach, describe, expect, it } from "vitest";
import { RuntimeRegistry } from "@dataforge/runtime";

const registry = new RuntimeRegistry();

afterEach(() => {
  registry.unregister("rt-a");
  registry.unregister("rt-b");
});

describe("RuntimeRegistry", () => {
  it("registers runtimes with capabilities and resources", () => {
    registry.register({
      runtimeId: "rt-a",
      capabilities: ["ray-data", "torch-npu"],
      resources: { cpu: 16, memoryGiB: 64, npu: 1 },
    });
    const entry = registry.get("rt-a");
    expect(entry?.status).toBe("registered");
    expect(entry?.resourceCapacity.npu).toBe(1);
  });

  it("selects the least loaded healthy runtime (§16/§22)", () => {
    registry.register({
      runtimeId: "rt-a",
      capabilities: ["torch-npu"],
      resources: { cpu: 16, memoryGiB: 64, npu: 1 },
    });
    registry.register({
      runtimeId: "rt-b",
      capabilities: ["torch-npu"],
      resources: { cpu: 16, memoryGiB: 64, npu: 1 },
    });
    registry.setActiveJobs("rt-a", 2);
    registry.setActiveJobs("rt-b", 0);

    const pick = registry.select({
      requiredCapabilities: ["torch-npu"],
      resources: { npu: 1 },
      healthyOnly: false,
    });
    expect(pick?.runtimeId).toBe("rt-b");
  });

  it("marks runtimes unhealthy after heartbeat timeout (§21)", () => {
    registry.register({
      runtimeId: "rt-a",
      capabilities: [],
      resources: { cpu: 1, memoryGiB: 1 },
    });
    // Force a stale heartbeat.
    const entry = registry.get("rt-a")!;
    entry.lastHeartbeatMs = Date.now() - 60_000;
    const marked = registry.applyHeartbeatTimeout(15_000);
    expect(marked).toContain("rt-a");
    expect(registry.get("rt-a")?.status).toBe("unhealthy");
  });

  it("heartbeat restores unhealthy runtimes", () => {
    registry.register({
      runtimeId: "rt-a",
      capabilities: [],
      resources: { cpu: 1, memoryGiB: 1 },
    });
    registry.setStatus("rt-a", "unhealthy");
    registry.heartbeat("rt-a", 0);
    expect(registry.get("rt-a")?.status).not.toBe("unhealthy");
  });
});
