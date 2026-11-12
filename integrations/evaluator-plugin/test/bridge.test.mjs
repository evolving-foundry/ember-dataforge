import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { callBridge, PROTOCOL } from "../dist/bridge.js";
import { parseConfig } from "../dist/index.js";

test("config rejects old database keys and incomplete content identities", () => {
  assert.throws(() => parseConfig({ manifest: "x", digest: "short" }));
  assert.throws(() => parseConfig({ manifest: "x", digest: "sha256:" + "a".repeat(64), storePath: "old.sqlite" }));
});

test("bridge cancellation waits for process exit", async () => {
  const directory = await mkdtemp(join(tmpdir(), "dd-bridge-test-"));
  const runner = join(directory, "runner.mjs");
  const pidfile = join(directory, "pid");
  await writeFile(runner, `import {writeFileSync} from 'node:fs'; writeFileSync(${JSON.stringify(pidfile)}, String(process.pid)); setInterval(()=>{},1000);`);
  // callBridge accepts an executable with -m MODULE COMMAND argv. This launcher
  // ignores those arguments and runs a Node fixture, using no Python install.
  const launcher = process.platform === "win32" ? undefined : join(directory, "launcher");
  if (!launcher) { await rm(directory, {recursive:true,force:true}); return; }
  await writeFile(launcher, `#!/bin/sh\nexec '${process.execPath}' '${runner}'\n`, {mode:0o755});
  const controller = new AbortController();
  const promise = callBridge(launcher, "verify", {}, controller.signal);
  let pid;
  for (let i=0;i<100;i++) { try { pid=Number(await readFile(pidfile,"utf8")); break; } catch { await new Promise(r=>setTimeout(r,20)); } }
  assert.ok(pid);
  controller.abort();
  await assert.rejects(promise, /cancelled/);
  assert.throws(() => process.kill(pid, 0));
  await rm(directory, {recursive:true,force:true});
});
