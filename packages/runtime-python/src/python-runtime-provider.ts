/**
 * PythonRuntimeProvider — long-lived Python Ray/ML Runtime (doc §3/§17).
 *
 * Spawns `python -m runtime.zmq_server` (a DEALER peer) and exposes it as
 * an ExecutionRuntime over the supervisor's ZmqTransport. The Python side
 * wraps the existing curator kernel without modifying it
 * (doc §41.2 Phase 1: zero intrusion).
 */

import { spawn, type ChildProcess } from "node:child_process";
import path from "node:path";
import type {
  DatasetMetadata,
  DatasetRef,
  ExecutionContext,
  ExecutionResult,
  OperationSpec,
  RuntimeCapability,
  RuntimeMessage,
} from "@dataforge/protocol";
import type { ZmqTransport, TransportError } from "@dataforge/transport-zmq";
import type { ExecutionRuntime, RuntimeProvider } from "@dataforge/runtime";

export interface PythonRuntimeOptions {
  /** Shared supervisor whose ROUTER the runtime connects to. */
  supervisor: {
    /** Endpoint of the ROUTER the spawned runtime must connect to. */
    readonly commandEndpoint: string;
    transport(runtimeId: string): ZmqTransport;
    registry: {
      register(payload: {
        runtimeId: string;
        capabilities: RuntimeCapability[];
        resources: { cpu: number; memoryGiB: number; npu?: number; gpu?: number };
        version?: string;
      }): unknown;
    };
  };
  runtimeId?: string;
  /** Python interpreter, defaults to the repo root .venv/bin/python. */
  pythonPath?: string;
  /** Directory whose `python/` subtree contains the `runtime` package. */
  projectRoot?: string;
  capabilities?: RuntimeCapability[];
  resources?: { cpu: number; memoryGiB: number; npu?: number; gpu?: number };
  /** Max concurrent jobs the runtime accepts (doc §22). */
  maxConcurrentJobs?: number;
  /** Extra env vars for the child process. */
  env?: Record<string, string>;
  /** Forward child stdout/stderr lines through this callback. */
  onLog?: (line: string) => void;
}

interface HelloResponse {
  runtimeId?: string;
  version?: string;
  capabilities?: RuntimeCapability[];
  resources?: { cpu: number; memoryGiB: number; npu?: number; gpu?: number };
}

interface ExecuteResponse {
  jobId: string;
  status?: string;
  output?: { datasetId?: string } & Record<string, unknown>;
  metrics?: Record<string, unknown>;
}

class PythonExecutionRuntime implements ExecutionRuntime {
  constructor(
    private readonly transport: ZmqTransport,
    private readonly runtimeId: string,
  ) {}

  async start(): Promise<void> {
    await this.transport.connect();
  }

  async execute(
    operation: OperationSpec,
    input?: DatasetRef,
    context?: ExecutionContext,
  ): Promise<ExecutionResult> {
    const payload = {
      operation: {
        name: operation.name,
        version: operation.version ?? "1",
        params: operation.params ?? {},
      },
      input: input ? { datasetId: input.id } : undefined,
      resources: context?.resources,
      awaitResult: context?.awaitResult ?? true,
      context: context?.metadata,
    };
    const response = await this.transport.request<unknown, ExecuteResponse>(
      "operation.execute",
      payload,
      {
        timeoutMs: context?.timeoutMs ?? 120_000,
        jobId: context?.jobId,
      },
    );
    return {
      jobId: response.jobId,
      output: response.output,
      status: (response.status as ExecutionResult["status"]) ?? "completed",
      metrics: response.metrics,
    };
  }

  async cancel(jobId: string): Promise<void> {
    await this.transport.request("job.cancel", { jobId }, { jobId });
  }

  async inspectDataset(ref: DatasetRef): Promise<DatasetMetadata> {
    return this.transport.request<unknown, DatasetMetadata>("dataset.inspect", {
      datasetId: ref.id,
    });
  }

  async releaseDataset(ref: DatasetRef): Promise<void> {
    await this.transport.request("dataset.release", { datasetId: ref.id });
  }

  async health(): Promise<{ runtimeId: string; healthy: boolean; details?: Record<string, unknown> }> {
    try {
      const payload = await this.transport.request<unknown, { runtimeId: string; activeJobs: number }>(
        "runtime.health",
        {},
        { runtimeId: this.runtimeId, timeoutMs: 10_000 },
      );
      return { runtimeId: payload.runtimeId ?? this.runtimeId, healthy: true, details: payload };
    } catch (error) {
      return {
        runtimeId: this.runtimeId,
        healthy: false,
        details: { error: error instanceof Error ? error.message : String(error) },
      };
    }
  }

  events(): AsyncIterable<RuntimeMessage> {
    return this.transport.events();
  }

  async dispose(): Promise<void> {
    await this.transport.close();
  }

  get id(): string {
    return this.runtimeId;
  }
}

export class PythonRuntimeProvider implements RuntimeProvider {
  readonly runtimeId: string;
  private readonly pythonPath: string;
  private readonly projectRoot: string;
  private readonly options: PythonRuntimeOptions;
  private child: ChildProcess | null = null;

  constructor(options: PythonRuntimeOptions) {
    this.options = options;
    this.runtimeId = options.runtimeId ?? `python-${Math.random().toString(36).slice(2, 8)}`;
    this.pythonPath = options.pythonPath ?? ".venv/bin/python";
    this.projectRoot = options.projectRoot ?? process.cwd();
  }

  /**
   * Spawn the Python runtime process and wait for its runtime.hello
   * handshake so callers can rely on routing being live afterwards.
   */
  async start(): Promise<ExecutionRuntime> {
    const root = this.projectRoot;
    const env: Record<string, string> = {
      ...process.env,
      PYTHONPATH: path.join(root, "python"),
      DATAFORGE_RUNTIME_ID: this.runtimeId,
      DATAFORGE_COMMAND_ENDPOINT: this.options.supervisor.commandEndpoint,
      DATAFORGE_MAX_CONCURRENT_JOBS: String(this.options.maxConcurrentJobs ?? 4),
      ...this.options.env,
    };

    this.child = spawn(this.pythonPath, ["-m", "curator.runtime.zmq_server"], {
      cwd: root,
      env,
      stdio: ["ignore", "pipe", "pipe"],
    });

    const log = this.options.onLog;
    if (log) {
      this.child.stdout?.on("data", (chunk: Buffer) => log(chunk.toString().trim()));
      this.child.stderr?.on("data", (chunk: Buffer) => log(chunk.toString().trim()));
    }
    this.child.on("exit", (code, signal) => {
      log?.(`[python-runtime ${this.runtimeId}] exited code=${code} signal=${signal}`);
    });

    const transport = this.options.supervisor.transport(this.runtimeId);
    const runtime = new PythonExecutionRuntime(transport, this.runtimeId);
    await runtime.start();

    // The Python runtime announces (runtime.hello) immediately on start;
    // wait for that first frame so the ROUTER→DEALER route is established
    // before the handshake request (ZeroMQ drops unrouted sends silently).
    await transport.waitForPeer(30_000);

    // Handshake: the first ROUTER→DEALER frames establish the identity in
    // both directions; runtime.hello doubles as capability registration.
    const hello = await transport.request<unknown, HelloResponse>(
      "runtime.hello",
      {
        runtimeId: this.runtimeId,
        capabilities: this.options.capabilities ?? ["ray-data"],
        resources: this.options.resources ?? { cpu: 8, memoryGiB: 16 },
      },
      { timeoutMs: 30_000 },
    );

    this.options.supervisor.registry.register({
      runtimeId: this.runtimeId,
      capabilities: hello.capabilities ?? this.options.capabilities ?? ["ray-data"],
      resources: hello.resources ?? this.options.resources ?? { cpu: 8, memoryGiB: 16 },
      version: hello.version,
    });

    return runtime;
  }

  async dispose(): Promise<void> {
    if (!this.child) return;
    const child = this.child;
    this.child = null;
    child.kill("SIGTERM");
    await new Promise<void>((resolve) => {
      const timer = setTimeout(() => {
        child.kill("SIGKILL");
        resolve();
      }, 5_000);
      child.once("exit", () => {
        clearTimeout(timer);
        resolve();
      });
    });
  }
}
