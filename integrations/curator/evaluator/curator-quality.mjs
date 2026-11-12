// curator-quality - thin Curator metrics plugin for Evaluator.
//
// It only assembles and forwards data: collect finalText from the run -> write a
// temporary JSONL file -> spawn `python -m curator.trusted_worker.op` with one
// JSON request/response over stdin/stdout -> read the metrics back. All
// computation happens on the Python side (Evaluator sections 4.1 and 4.3).
//
// Loading: cordis.patch.yaml in the same directory as the experiment YAML
//   - insert:
//       - id: curator-quality
//         name: ./plugins/curator-quality.mjs
// Experiment YAML:
//   metrics:
//     - name: curator-quality
//       config:
//         pythonExecutable: <venv>/bin/python
//         op: CandidateLogDeduplicate      # or CandidateBatchQualityAssess
//         # includeStatuses: [completed]   # defaults to completed only
//         # params: {}                     # pass operator parameters through
//         # env: { PYTHONPATH: <curator repo root or venv> }
//         # timeoutMs: 120000
//
// Invariant (section 4.5): metric failure means report failure (throw -> run
// exits with code 1 and produces no metrics.json). Empty input short-circuits
// without spawning, and failures never produce fabricated metrics.
import { spawn } from "node:child_process";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

const SCHEMA_VERSION = 1;
const SUPPORTED_OPS = new Set(["CandidateLogDeduplicate", "CandidateBatchQualityAssess"]);
const ALL_STATUSES = new Set(["completed", "error", "timeout", "cancelled", "unsupported"]);
const CONFIG_KEYS = new Set(["pythonExecutable", "op", "includeStatuses", "params", "env", "timeoutMs"]);
const DEFAULT_TIMEOUT_MS = 120_000;

function rejectUnknown(record, allowed, where) {
  const unknown = Object.keys(record).filter(key => !allowed.has(key));
  if (unknown.length > 0) throw new TypeError(`${where} has unknown keys: ${unknown.sort().join(", ")}`);
}

function expectString(value, where) {
  if (typeof value !== "string" || value.trim() === "") throw new TypeError(`${where} must be a non-empty string`);
  return value;
}

function validate(config) {
  if (config === null || typeof config !== "object" || Array.isArray(config)) {
    throw new TypeError("plugin config must be an object");
  }
  rejectUnknown(config, CONFIG_KEYS, "plugin config");
  const op = expectString(config.op, "config.op");
  if (!SUPPORTED_OPS.has(op)) {
    throw new TypeError(`config.op must be one of ${[...SUPPORTED_OPS].sort().join(", ")}`);
  }
  let includeStatuses = ["completed"];
  if (config.includeStatuses !== undefined) {
    if (!Array.isArray(config.includeStatuses) || config.includeStatuses.some(s => !ALL_STATUSES.has(s))) {
      throw new TypeError(`config.includeStatuses must be a list of ${[...ALL_STATUSES].sort().join(", ")}`);
    }
    includeStatuses = [...new Set(config.includeStatuses)];
  }
  if (config.params !== undefined && (config.params === null || typeof config.params !== "object" || Array.isArray(config.params))) {
    throw new TypeError("config.params must be an object");
  }
  if (config.env !== undefined && (config.env === null || typeof config.env !== "object" || Array.isArray(config.env)
    || Object.values(config.env).some(v => typeof v !== "string"))) {
    throw new TypeError("config.env must be a record of strings");
  }
  if (config.timeoutMs !== undefined && (typeof config.timeoutMs !== "number" || !Number.isFinite(config.timeoutMs) || config.timeoutMs <= 0)) {
    throw new TypeError("config.timeoutMs must be a positive finite number");
  }
  if (config.pythonExecutable !== undefined) expectString(config.pythonExecutable, "config.pythonExecutable");
  return { ...config, includeStatuses, params: config.params ?? {}, env: config.env ?? {} };
}

function eligibleRecords(snapshot, includeStatuses) {
  return snapshot.records.filter(record =>
    includeStatuses.includes(record.status)
    && record.solve !== null
    && typeof record.solve.finalText === "string");
}

function runWorker({ pythonExecutable, request, env, timeoutMs }) {
  return new Promise((resolve, reject) => {
    const child = spawn(pythonExecutable, ["-m", "curator.trusted_worker.op"], {
      env: { ...process.env, ...env },
      stdio: ["pipe", "pipe", "pipe"],
    });
    let stdout = "";
    let stderr = "";
    let settled = false;
    const settle = (error, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (error) reject(error); else resolve(value);
    };
    const timer = setTimeout(() => {
      child.kill("SIGTERM");
      settle(new Error(`curator-quality worker timed out after ${timeoutMs}ms`));
    }, timeoutMs);
    child.on("error", error => settle(error));
    child.stdout.on("data", chunk => { stdout += chunk; });
    child.stderr.on("data", chunk => { stderr += chunk; });
    child.on("close", code => {
      if (code !== 0) {
        settle(new Error(`curator-quality worker exited ${code}: ${stderr.trim() || "no diagnostics"}`));
        return;
      }
      try {
        settle(undefined, JSON.parse(stdout));
      } catch (error) {
        settle(new Error(`curator-quality worker produced unparsable stdout: ${error.message}`));
      }
    });
    child.stdin.on("error", error => settle(error));
    child.stdin.write(JSON.stringify(request) + "\n");
    child.stdin.end();
  });
}

export const inject = ["evaluator"];
export function apply(ctx) {
  ctx.evaluator.register("metrics", "curator-quality", validate, (config, runtime) => ({
    name: "curator-quality",
    async compute(snapshot) {
      const records = eligibleRecords(snapshot, config.includeStatuses);
      // Empty-input short circuit (section 4.3): do not spawn when there are no
      // eligible records, avoiding a meaningless failure for an empty run.
      if (records.length === 0) return { eligible: 0 };

      const pythonExecutable = config.pythonExecutable === undefined ? "python"
        : (config.pythonExecutable.includes("/") || config.pythonExecutable.includes("\\")
          ? runtime.resolvePath(config.pythonExecutable)
          : config.pythonExecutable);
      const workspace = await mkdtemp(join(tmpdir(), "curator-quality-"));
      try {
        const dataPath = join(workspace, "final-texts.jsonl");
        await writeFile(
          dataPath,
          records.map(record => JSON.stringify({ taskId: record.key.taskId, text: record.solve.finalText })).join("\n") + "\n",
          "utf8",
        );
        const response = await runWorker({
          pythonExecutable,
          request: {
            schemaVersion: SCHEMA_VERSION,
            op: config.op,
            params: config.params,
            dataPath,
            outputDir: workspace,
          },
          env: config.env,
          timeoutMs: config.timeoutMs ?? DEFAULT_TIMEOUT_MS,
        });
        if (response.schemaVersion !== SCHEMA_VERSION || response.op !== config.op
          || response.metrics === null || typeof response.metrics !== "object" || Array.isArray(response.metrics)) {
          throw new Error("curator-quality worker response violates the op-mode contract");
        }
        return response.metrics;
      } finally {
        await rm(workspace, { recursive: true, force: true }).catch(() => { /* Temporary directory cleanup failure does not affect metrics. */ });
      }
    },
  }));
}
export default { inject, apply };
