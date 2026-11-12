import { mkdtemp, mkdir, rm, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, posix } from "node:path";
import type { Context } from "@deepseek-ai/cordis";
import type { Benchmark, EnvironmentSession, JsonValue, PreparedTask, SolveResult, SourceLock, TaskRef, VerifyResult } from "ember-evaluator";
import { callBridge, PROTOCOL } from "./bridge.js";

export type Config = { manifest: string; digest: string; pythonExecutable: string; timeoutSeconds: number };
type Entry = { id: string; recordKey: number; public: { task_id: string; task_text: string };
  deliverables: string[]; attachments: { source: string; target: string }[]; textOnly: boolean };

export function parseConfig(value: unknown): Config {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new TypeError("benchmark config must be an object");
  const record = value as Record<string, unknown>;
  for (const key of Object.keys(record)) if (!["manifest", "digest", "pythonExecutable", "timeoutSeconds"].includes(key)) throw new TypeError(`unknown benchmark config: ${key}`);
  if (typeof record.manifest !== "string" || !record.manifest) throw new TypeError("manifest is required");
  if (typeof record.digest !== "string" || !/^sha256:[0-9a-f]{64}$/.test(record.digest)) throw new TypeError("complete SHA-256 digest is required");
  const pythonExecutable = record.pythonExecutable ?? "python";
  const timeoutSeconds = record.timeoutSeconds ?? 60;
  if (typeof pythonExecutable !== "string" || !pythonExecutable) throw new TypeError("pythonExecutable must be a string");
  if (typeof timeoutSeconds !== "number" || !Number.isFinite(timeoutSeconds) || timeoutSeconds <= 0) throw new TypeError("timeoutSeconds must be positive");
  return { manifest: record.manifest, digest: record.digest, pythonExecutable, timeoutSeconds };
}

export class DataForgeBenchmark implements Benchmark {
  readonly config: Config;
  private items: Entry[] | undefined;
  constructor(config: Config) { this.config = config; }
  private call(command: string, request: Record<string, unknown>, signal: AbortSignal) {
    return callBridge(this.config.pythonExecutable, command, { protocol_version: PROTOCOL,
      manifest: this.config.manifest, digest: this.config.digest, ...request }, signal, this.config.timeoutSeconds * 1000);
  }
  async resolveSource(previous: SourceLock | null, signal: AbortSignal): Promise<SourceLock> {
    this.items = undefined;
    const source = await this.call("resolve-source", { previous }, signal) as SourceLock;
    if (source?.revision !== this.config.digest || source.source !== "dataforge-benchmark") throw new Error("invalid benchmark source lock");
    return source;
  }
  private async load(source: SourceLock, signal: AbortSignal): Promise<Entry[]> {
    signal.throwIfAborted();
    if (source.revision !== this.config.digest) throw new Error("benchmark source lock drift");
    if (!this.items) {
      const value = await this.call("enumerate", {}, signal) as { items: Entry[] };
      if (!Array.isArray(value?.items) || !value.items.length) throw new Error("empty benchmark selection");
      const ids = new Set<string>();
      for (const item of value.items) {
        if (typeof item.id !== "string" || ids.has(item.id) || !Number.isSafeInteger(item.recordKey) ||
          typeof item.public?.task_text !== "string" || item.public.task_id !== item.id || !Array.isArray(item.deliverables) || !item.deliverables.length) throw new Error("invalid benchmark item");
        for (const name of item.deliverables) if (typeof name !== "string" || name.startsWith("/") || name.includes("\\") || name.split("/").includes("..")) throw new Error("invalid public deliverable path");
        ids.add(item.id);
      }
      this.items = value.items;
    }
    return this.items;
  }
  async *enumerate(source: SourceLock, signal: AbortSignal): AsyncIterable<TaskRef> {
    for (const item of await this.load(source, signal)) { signal.throwIfAborted(); yield { id: item.id, recordKey: item.recordKey }; }
  }
  async prepare(ref: TaskRef, source: SourceLock, signal: AbortSignal): Promise<PreparedTask> {
    const item = (await this.load(source, signal)).find(item => item.id === ref.id && item.recordKey === ref.recordKey);
    if (!item) throw new Error("task reference is outside the locked selection");
    return { input: { id: item.id, prompt: item.public.task_text },
      verification: { taskId: item.id, digest: this.config.digest, deliverables: item.deliverables, textOnly: item.textOnly },
      solveEnv: { workdir: "/workspace", image: null, initialFiles: item.attachments, runtime: [] },
      verifyEnv: null, transfer: null, phaseTimeoutSeconds: { verify: this.config.timeoutSeconds } };
  }
  async verify(task: PreparedTask, result: SolveResult, session: EnvironmentSession | null, signal: AbortSignal): Promise<VerifyResult> {
    signal.throwIfAborted();
    const verification = task.verification as { taskId: string; digest: string; deliverables: string[]; textOnly: boolean };
    if (!verification || verification.digest !== this.config.digest || verification.taskId !== task.input.id) throw new Error("verification task identity mismatch");
    let root: string | undefined;
    try {
      if (!verification.textOnly) {
        if (!session) throw new Error("artifact benchmark requires a host environment session");
        root = await mkdtemp(join(tmpdir(), "dataforge-collected-"));
        for (const relative of verification.deliverables) {
          const target = join(root, relative);
          await mkdir(join(target, ".."), { recursive: true });
          await session.download(posix.join(session.workdir, relative), target, signal);
          if ((await stat(target)).size > 32 * 1024 * 1024) throw new Error("candidate artifact exceeds collection limit");
        }
      }
      const verdict = await this.call("verify", { task_id: verification.taskId,
        ...(root ? { candidate_root: root } : { final_text: result.finalText }), timeout_seconds: this.config.timeoutSeconds }, signal) as VerifyResult;
      if (typeof verdict.passed !== "boolean" || typeof verdict.score !== "number" || !Number.isFinite(verdict.score) || verdict.score < 0 || verdict.score > 1) throw new Error("invalid verifier result; no score recorded");
      return verdict;
    } finally { if (root) await rm(root, { recursive: true, force: true }); }
  }
}

export const inject = ["evaluator"];
export function apply(ctx: Context): void {
  ctx.evaluator.register("benchmark", "dataforge", parseConfig,
    (config, runtime) => new DataForgeBenchmark({ ...config, manifest: runtime.resolvePath(config.manifest) }));
}
export default { inject, apply };
