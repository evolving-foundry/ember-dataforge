/**
 * RuntimeRegistry (doc §15) — TypeScript-side bookkeeping of every known
 * runtime: identity, status, capabilities, resource capacity, active jobs,
 * heartbeat and version. The scheduler queries it to pick a runtime (§16)
 * and applies the capacity model for backpressure (§22).
 */

import type { RuntimeCapability } from "@dataforge/protocol";

export type RuntimeStatus = "registered" | "healthy" | "unhealthy" | "stopped";

export interface RuntimeResourceCapacity {
  cpu: number;
  memoryGiB: number;
  npu?: number;
  gpu?: number;
}

export interface RuntimeRegistryEntry {
  runtimeId: string;
  status: RuntimeStatus;
  capabilities: RuntimeCapability[];
  resourceCapacity: RuntimeResourceCapacity;
  activeJobs: number;
  /** Current queue depth as reported by the runtime (§22). */
  queueDepth: number;
  device?: string;
  version?: string;
  /** Unix epoch ms of the last heartbeat/register frame. */
  lastHeartbeatMs?: number;
}

export interface RuntimeSelectionQuery {
  requiredCapabilities?: RuntimeCapability[];
  resources?: { npu?: number; gpu?: number; cpu?: number };
  /** When true, only runtimes with healthy heartbeat are considered. */
  healthyOnly?: boolean;
}

export class RuntimeRegistry {
  private readonly entries = new Map<string, RuntimeRegistryEntry>();

  register(payload: {
    runtimeId: string;
    capabilities: RuntimeCapability[];
    resources: RuntimeResourceCapacity;
    version?: string;
    device?: string;
  }): RuntimeRegistryEntry {
    const existing = this.entries.get(payload.runtimeId);
    const entry: RuntimeRegistryEntry = {
      runtimeId: payload.runtimeId,
      status: "registered",
      capabilities: payload.capabilities,
      resourceCapacity: payload.resources,
      activeJobs: existing?.activeJobs ?? 0,
      queueDepth: existing?.queueDepth ?? 0,
      device: payload.device ?? existing?.device,
      version: payload.version ?? existing?.version,
      lastHeartbeatMs: Date.now(),
    };
    this.entries.set(payload.runtimeId, entry);
    return entry;
  }

  get(runtimeId: string): RuntimeRegistryEntry | undefined {
    return this.entries.get(runtimeId);
  }

  list(): RuntimeRegistryEntry[] {
    return [...this.entries.values()];
  }

  heartbeat(runtimeId: string, activeJobs?: number, queueDepth?: number): void {
    const entry = this.entries.get(runtimeId);
    if (!entry) return;
    entry.lastHeartbeatMs = Date.now();
    if (entry.status === "unhealthy") entry.status = "healthy";
    if (activeJobs !== undefined) entry.activeJobs = activeJobs;
    if (queueDepth !== undefined) entry.queueDepth = queueDepth;
  }

  setStatus(runtimeId: string, status: RuntimeStatus): void {
    const entry = this.entries.get(runtimeId);
    if (entry) entry.status = status;
  }

  setActiveJobs(runtimeId: string, activeJobs: number): void {
    const entry = this.entries.get(runtimeId);
    if (entry) entry.activeJobs = activeJobs;
  }

  unregister(runtimeId: string): void {
    this.entries.delete(runtimeId);
  }

  /** Mark runtimes whose heartbeat is older than timeoutMs as unhealthy. */
  applyHeartbeatTimeout(timeoutMs: number): string[] {
    const marked: string[] = [];
    const now = Date.now();
    for (const entry of this.entries.values()) {
      if (
        entry.status !== "stopped" &&
        entry.lastHeartbeatMs !== undefined &&
        now - entry.lastHeartbeatMs > timeoutMs
      ) {
        if (entry.status !== "unhealthy") marked.push(entry.runtimeId);
        entry.status = "unhealthy";
      }
    }
    return marked;
  }

  /**
   * Scheduler query (doc §16/§22): pick a runtime that has the required
   * capabilities, satisfies resource requirements, and is healthy.
   */
  select(query: RuntimeSelectionQuery = {}): RuntimeRegistryEntry | undefined {
    const candidates = this.list().filter((entry) => {
      if (query.healthyOnly && entry.status !== "healthy" && entry.status !== "registered") {
        return false;
      }
      if (query.requiredCapabilities?.length) {
        const has = query.requiredCapabilities.every((cap) =>
          entry.capabilities.includes(cap),
        );
        if (!has) return false;
      }
      const res = query.resources;
      if (res) {
        if (res.cpu !== undefined && entry.resourceCapacity.cpu < res.cpu) return false;
        if (res.npu !== undefined && (entry.resourceCapacity.npu ?? 0) < res.npu) return false;
        if (res.gpu !== undefined && (entry.resourceCapacity.gpu ?? 0) < res.gpu) return false;
      }
      return true;
    });
    if (candidates.length === 0) return undefined;
    // Prefer the least loaded (activeJobs, then queueDepth).
    candidates.sort(
      (a, b) => a.activeJobs - b.activeJobs || a.queueDepth - b.queueDepth,
    );
    return candidates[0];
  }
}
