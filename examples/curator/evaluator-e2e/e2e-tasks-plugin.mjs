// Minimal benchmark and solver for the end-to-end test. The solver echoes
// preset text for each task, with deliberate duplicates that exercise
// curator-quality CandidateLogDeduplicate (six finalText values:
// alpha/beta/alpha/gamma/beta/delta -> kept 4, removed 2).
// No network requests are made; the model config only satisfies the schema,
// and the solver does not call a model.
import { writeFile } from "node:fs/promises";
import { join } from "node:path";

const TASK_TEXTS = [
  ["t1", "alpha solution text"],
  ["t2", "beta plan text"],
  ["t3", "alpha solution text"],
  ["t4", "gamma review text"],
  ["t5", "beta plan text"],
  ["t6", "delta notes text"],
];

const config = (value) => value;

export const inject = ["evaluator"];
export function apply(ctx) {
  ctx.evaluator.register("benchmark", "e2e-tasks", config, () => ({
    async resolveSource() {
      return { source: "e2e-tasks", revision: "fixed", files: [], metadata: {} };
    },
    async *enumerate() {
      for (const [id] of TASK_TEXTS) yield { id, recordKey: id };
    },
    async prepare(task) {
      const text = TASK_TEXTS.find(([id]) => id === task.id)?.[1] ?? "";
      return {
        input: { id: task.id, prompt: text },
        verification: null,
        solveEnv: { workdir: ".", image: null, initialFiles: [], runtime: [] },
        verifyEnv: null,
        transfer: null,
      };
    },
    async verify(_task, result) {
      const passed = typeof result.finalText === "string" && result.finalText.length > 0;
      return { score: Number(passed), passed, details: {} };
    },
  }));
  ctx.evaluator.register("solver", "e2e-echo", config, () => ({
    requirements() {
      return [];
    },
    async solve(input, _session, context) {
      const event = (seq, type) =>
        JSON.stringify({
          schemaVersion: 1,
          key: context.key,
          seq,
          type,
          at: new Date().toISOString(),
          data: null,
        }) + "\n";
      await writeFile(
        join(context.paths.agent, "trajectory.jsonl"),
        event(1, "start") + event(2, "end"),
      );
      await writeFile(join(context.paths.agent, "session.jsonl"), '{"e2e":true}\n');
      return {
        finalText: input.prompt,
        usage: { inputTokens: null, outputTokens: null, totalTokens: null },
      };
    },
  }));
}
