import { spawn } from "node:child_process";

export const PROTOCOL = "dataforge-evaluator-v1";

/** Asynchronous and bounded. Abort settles only after the child has exited. */
export function callBridge(python: string, command: string, request: unknown, signal: AbortSignal,
  timeoutMs = 60_000, module = "data_designer.integrations.evaluator.bridge"): Promise<unknown> {
  signal.throwIfAborted();
  return new Promise((resolve, reject) => {
    const child = spawn(python, ["-m", module, command], {
      stdio: ["pipe", "pipe", "pipe"], windowsHide: true, detached: process.platform !== "win32",
    });
    let stdout = Buffer.alloc(0);
    let stderrBytes = 0;
    let failure: Error | undefined;
    let terminating: Promise<void> | undefined;
    const stop = (error: Error) => {
      failure ??= error;
      if (terminating || !child.pid) return;
      if (process.platform === "win32") {
        terminating = new Promise<void>(done => {
          const killer = spawn("taskkill", ["/PID", String(child.pid), "/T", "/F"], { windowsHide: true, stdio: "ignore" });
          killer.once("error", () => { child.kill(); done(); });
          killer.once("close", () => done());
        });
      } else {
        try { process.kill(-child.pid, "SIGKILL"); } catch { child.kill("SIGKILL"); }
        terminating = Promise.resolve();
      }
    };
    const onAbort = () => stop(new Error("Evaluator bridge cancelled"));
    const timer = setTimeout(() => stop(new Error("Evaluator bridge timed out")), timeoutMs);
    signal.addEventListener("abort", onAbort, { once: true });
    child.stdout.on("data", (chunk: Buffer) => {
      if (stdout.length + chunk.length > 8 * 1024 * 1024) stop(new Error("Evaluator bridge stdout limit exceeded"));
      else stdout = Buffer.concat([stdout, chunk]);
    });
    child.stderr.on("data", (chunk: Buffer) => {
      stderrBytes += chunk.length;
      if (stderrBytes > 1024 * 1024) stop(new Error("Evaluator bridge diagnostic limit exceeded"));
    });
    child.stdin.on("error", () => { /* Exit status is handled by close. */ });
    child.once("error", error => { failure ??= error; });
    child.once("close", async code => {
      clearTimeout(timer);
      signal.removeEventListener("abort", onAbort);
      await terminating;
      if (failure) return reject(failure);
      if (code !== 0) return reject(new Error(`Evaluator bridge failed (exit ${code})`));
      try {
        const envelope = JSON.parse(stdout.toString("utf8")) as Record<string, unknown>;
        if (envelope.protocol_version !== PROTOCOL || !("result" in envelope)) throw new Error("Unsupported evaluator bridge response");
        resolve(envelope.result);
      } catch (error) { reject(error); }
    });
    child.stdin.end(JSON.stringify(request));
    if (signal.aborted) onAbort();
  });
}
