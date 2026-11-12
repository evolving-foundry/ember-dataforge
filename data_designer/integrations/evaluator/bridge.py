"""One JSON request/response per invocation; no database or generation imports."""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from data_designer.pipelines.benchmarks.reader import enumerate_public, read_package, resolve_resources

PROTOCOL = "dataforge-evaluator-v1"


def deliverables(task):
    spec = task["private"].get("private", {}).get("verifier_spec", {})
    payload = spec.get("payload", {})
    artifact = payload.get("artifact", {})
    relative = artifact.get("path") or payload.get("deliverable")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError("family has no expressible, clean relative deliverable")
    return [relative]


def handle(command, request):
    if request.get("protocol_version") != PROTOCOL:
        raise ValueError("unsupported evaluator bridge protocol")
    package = read_package(request["manifest"], expected_digest=request["digest"])
    if command == "resolve-source":
        source = {"source": "dataforge-benchmark", "revision": package["digest"],
                  "files": [{"path": "manifest.json", "sha256": package["digest"].split(":")[1], "origin": "dataforge-benchmark-content-v1"}],
                  "metadata": {"digest": package["digest"], "taskCount": len(package["content"]["tasks"])}}
        if request.get("previous") is not None and request["previous"] != source:
            raise ValueError("previous source lock differs from the frozen benchmark")
        return source
    if command == "enumerate":
        output = []
        for index, task in enumerate(package["content"]["tasks"]):
            public = task["public"]
            output.append({"recordKey": index, "id": public["task_id"], "public": public,
                           "deliverables": deliverables(task),
                           "attachments": resolve_resources(task.get("attachments", []), package),
                           "textOnly": public["benchmark"]["family"] in {"reasoning_qa", "knowledge_qa", "code_gen", "health_qa"}})
        return {"items": output}
    if command != "verify":
        raise ValueError("unknown evaluator bridge command")
    from data_designer.pipelines.benchmarks.verification import verify_task
    task_id = request["task_id"]
    candidate_root = request.get("candidate_root")
    if candidate_root:
        verdict = verify_task(package["manifest_uri"], package["digest"], task_id, candidate_root,
                              timeout=request.get("timeout_seconds", 60))
    else:
        tasks = [task for task in package["content"]["tasks"] if task["public"]["task_id"] == task_id]
        if len(tasks) != 1 or tasks[0]["public"]["benchmark"]["family"] not in {"reasoning_qa", "knowledge_qa", "code_gen", "health_qa"}:
            raise ValueError("artifact tasks require host-collected candidate files")
        with tempfile.TemporaryDirectory(prefix="dataforge-answer-") as temporary:
            path = Path(temporary) / deliverables(tasks[0])[0]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(request["final_text"], encoding="utf-8")
            verdict = verify_task(package["manifest_uri"], package["digest"], task_id, temporary,
                                  timeout=request.get("timeout_seconds", 60))
    return {"passed": verdict["passed"], "score": verdict["score"],
            "details": {"status": verdict["status"], "verifierVersion": verdict["verifier_version"],
                        "evidence": verdict["evidence_refs"], "sourceDigest": package["digest"], "taskId": task_id}}


def main(argv=None):
    args = sys.argv[1:] if argv is None else argv
    try:
        if len(args) != 1:
            raise ValueError("one bridge command is required")
        raw = sys.stdin.buffer.read(1024 * 1024 + 1)
        if len(raw) > 1024 * 1024:
            raise ValueError("bridge input exceeds size limit")
        result = handle(args[0], json.loads(raw))
        print(json.dumps({"protocol_version": PROTOCOL, "result": result}, ensure_ascii=False, allow_nan=False))
        return 0
    except Exception as error:
        # Error messages from provider/host exceptions are not safe to echo verbatim.
        print(f"evaluator bridge {type(error).__name__}: request could not complete", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
