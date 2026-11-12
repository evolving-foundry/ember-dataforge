#!/usr/bin/env python3
"""Formal Curator -> DataForge -> Gym environment-synthesis example.

The example intentionally uses real DOCX and PDF files.  It exercises the
same contracts as production:

1. Curator parses the files into ``parsed-doc-v2`` rows.
2. ``envsynth`` source-import worker writes a seed-import bundle.
3. ``env_synthesis_worker`` freezes ``env_release`` and ``taskset`` into the
   artifact store and writes one Gym-native package (``manifest.json`` /
   ``env_spec.json`` / ``tasks.jsonl`` / office assets).
4. ``scripts/gym_publish_env.py`` places that package plus the benchmark shell
   into a Gym overlay; package bytes and fingerprints are never rewritten.
5. When a Gym checkout with ``resources_servers/synth_env`` is discoverable
   (``--gym-root`` / ``DATA_FORGE_GYM`` / sibling checkout), the example also
   loads the published zip through Gym's own ``load_package_zip`` and executes
   every task against the office profile.

Generated files are kept so the result can be inspected.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from data_designer.pipelines.envsynth.serialization import json_bytes


REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLISH_SCRIPT = REPO_ROOT / "scripts" / "gym_publish_env.py"
SUMMARY_SCHEMA = "envsynth-curator-e2e-example-v1"
SUMMARY_NAME = "example_summary.json"
REF_ID = "curator-docx-pdf-ocr-e2e"
BENCHMARK_NAME = "curator-docx-pdf-ocr-e2e"

TITLE = "电力工单台账"
HEADERS = ("Ticket ID", "标题", "状态")
TICKETS = (
    ("GD-2026-2001", "线路检修", "已审批"),
    ("GD-2026-2002", "缺陷处理", "审核中"),
)
REGULATION_LINES = (
    "Work order regulation",
    "Every work order requires two-person approval before closure.",
    "Rejected work orders return to draft status for revision.",
)
OCR_LINES = (
    "Power Work Order",
    "ticket id: GD-2026-2003",
    "title: OCRInspection",
    "status: approved",
)


class ExampleError(RuntimeError):
    """The example cannot continue or an expected invariant failed."""


def _require_runtime(ocr_backend: str = "paddleocr") -> None:
    requirements = {
        "docx": "python-docx",
        "pdfplumber": "pdfplumber",
        "jsonschema": "jsonschema",
        "PIL": "Pillow",
        "curator": "curator",
    }
    if ocr_backend == "paddleocr":
        requirements["paddleocr"] = "paddleocr"
    elif ocr_backend == "mindocr":
        requirements["mindocr"] = "mindocr"
    elif ocr_backend == "auto":
        if (
            importlib.util.find_spec("paddleocr") is None
            and importlib.util.find_spec("mindocr") is None
        ):
            requirements["paddleocr | mindocr"] = "PaddleOCR or MindOCR"
    else:
        raise ExampleError("Unknown OCR backend: {}".format(ocr_backend))
    missing = [
        package
        for module, package in requirements.items()
        if importlib.util.find_spec(module) is None
    ]
    if missing:
        raise ExampleError(
            "Missing example dependencies: {}. Install from the repository root with `pip install -e .[docs,ocr-paddleocr]`.".format(
                ", ".join(missing)
            )
        )


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(json_bytes(value))


def _run(command: List[str], *, stdin: Optional[str] = None) -> subprocess.CompletedProcess:
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise ExampleError(
            "Command failed (exit={}): {}\nstdout:\n{}\nstderr:\n{}".format(
                result.returncode,
                " ".join(command),
                result.stdout.strip(),
                result.stderr.strip(),
            )
        )
    return result


def _last_json(stdout: str, *, label: str) -> Dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ExampleError("{} produced no JSON output".format(label))
    try:
        value = json.loads(lines[-1])
    except ValueError as error:
        raise ExampleError("{} output is not JSON: {}".format(label, error)) from error
    if not isinstance(value, dict):
        raise ExampleError("{} output must be a JSON object".format(label))
    return value


def _prepare_workspace(work_dir: Optional[str]) -> Path:
    if work_dir is None:
        return Path(tempfile.mkdtemp(prefix="envsynth-curator-e2e-")).resolve()
    root = Path(work_dir).expanduser().resolve()
    if root.exists():
        raise ExampleError("--work-dir must be a directory that does not exist yet: {}".format(root))
    root.mkdir(parents=True)
    return root


def _write_docx(path: Path) -> None:
    import docx

    document = docx.Document()
    document.add_paragraph(TITLE)
    table = document.add_table(rows=len(TICKETS) + 1, cols=len(HEADERS))
    for col, text in enumerate(HEADERS):
        table.cell(0, col).text = text
    for row_index, ticket in enumerate(TICKETS, start=1):
        for col, text in enumerate(ticket):
            table.cell(row_index, col).text = text
    document.save(str(path))


def _write_pdf(path: Path) -> None:
    content = ["BT", "/F1 12 Tf", "72 720 Td"]
    for index, line in enumerate(REGULATION_LINES):
        escaped = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        if index:
            content.append("0 -18 Td")
        content.append("({}) Tj".format(escaped))
    content.append("ET")
    stream = ("\n".join(content) + "\n").encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>"
        ),
        (
            b"<< /Length "
            + str(len(stream)).encode("ascii")
            + b" >>\nstream\n"
            + stream
            + b"endstream"
        ),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    header = b"%PDF-1.4\n"
    body = b""
    offsets = []
    for number, value in enumerate(objects, start=1):
        offsets.append(len(header) + len(body))
        body += "{} 0 obj\n".format(number).encode("ascii") + value + b"\nendobj\n"
    xref_offset = len(header) + len(body)
    xref = "xref\n0 {}\n0000000000 65535 f \n".format(len(objects) + 1).encode("ascii")
    xref += b"".join(
        "{:010d} 00000 n \n".format(offset).encode("ascii") for offset in offsets
    )
    trailer = (
        "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n".format(
            len(objects) + 1, xref_offset
        ).encode("ascii")
    )
    path.write_bytes(header + body + xref + trailer)


def _write_png(path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1200, 500), "white")
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 42)
    except OSError:
        font = ImageFont.load_default()
    for index, line in enumerate(OCR_LINES):
        draw.text((60, 50 + index * 85), line, fill="black", font=font)
    image.save(str(path), format="PNG")


def _source_input(
    docx_path: Path,
    pdf_path: Path,
    png_path: Path,
    *,
    ocr_backend: str,
    ocr_device: str,
    ocr_lang: str,
) -> Dict[str, Any]:
    return {
        "schemaVersion": 1,
        "request": {
            "sourcePaths": [str(docx_path), str(png_path)],
            "regulationPaths": [str(pdf_path)],
            "family": "power-ticket",
            "anchorHoldoutRatio": 0.3,
            "maxQuestions": 20,
            "ocrOptions": {
                "backend": ocr_backend,
                "device": ocr_device,
                "lang": ocr_lang,
                "layout_analysis": True,
                "allow_model_download": True,
            },
        },
    }


def _compile_input(
    *,
    asset_root: Path,
    store_path: Path,
    seed_manifest_path: Path,
    seed_manifest_sha256: str,
) -> Dict[str, Any]:
    return {
        "schemaVersion": 1,
        "request": {
            "storePath": str(store_path),
            "assetRoot": str(asset_root),
            "refId": REF_ID,
            "family": "power-ticket",
            "split": "train",
            "seedManifestPath": str(seed_manifest_path),
            "seedManifestSha256": seed_manifest_sha256,
            "envProfile": "office",
            "synthesis": {
                "mode": "template",
                "maxRounds": 3,
                "grounding": {"embedModel": "char-ngram-v1", "topK": 4},
            },
            "curationAnswersPath": None,
            "qualityThreshold": 0.5,
            "anchorHoldoutRatio": 0.3,
            "expertReview": {"mode": "waived", "sampleRate": 0.0},
            "seed": 42,
        },
    }


def _verify_store(store_path: Path, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    from data_designer.storage.catalog import ArtifactCatalog

    release = snapshot["envRelease"]
    taskset = snapshot["taskset"]
    store = ArtifactCatalog(str(store_path))
    try:
        release_artifact = store.get_artifact(
            "env_release", str(release["refId"]), int(release["version"])
        )
        taskset_artifact = store.get_artifact(
            "taskset", str(taskset["refId"]), int(taskset["version"])
        )
    finally:
        store.close()
    if release_artifact is None:
        raise ExampleError("Artifact directory is missing env_release/{}/v{}".format(
            release["refId"], release["version"]
        ))
    if taskset_artifact is None:
        raise ExampleError("Artifact directory is missing taskset/{}/v{}".format(
            taskset["refId"], taskset["version"]
        ))
    if release_artifact["fingerprint"] != release["fingerprint"]:
        raise ExampleError("Artifact directory env_release fingerprint does not match snapshot")
    if taskset_artifact["fingerprint"] != taskset["fingerprint"]:
        raise ExampleError("Artifact directory taskset fingerprint does not match snapshot")
    return {
        "releaseFingerprint": release_artifact["fingerprint"],
        "tasksetFingerprint": taskset_artifact["fingerprint"],
    }


def _discover_gym_root(explicit: Optional[str]) -> Optional[Path]:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser().resolve())
    env_root = os.environ.get("DATA_FORGE_GYM")
    if env_root:
        candidates.append(Path(env_root).expanduser().resolve())
    candidates.append((REPO_ROOT.parent / "AscendAgent-Gym").resolve())
    for candidate in candidates:
        if (candidate / "overlay/resources_servers/synth_env").is_dir():
            return candidate
    if explicit:
        raise ExampleError(
            "--gym-root is not a Gym checkout containing overlay/resources_servers/synth_env: {}".format(
                explicit
            )
        )
    return None


def _load_gym_runtime(gym_root: Path):
    overlay = str(gym_root / "overlay")
    if overlay not in sys.path:
        sys.path.insert(0, overlay)
    from resources_servers.synth_env.engine.events import EventScheduler
    from resources_servers.synth_env.engine.npc import NpcEngine
    from resources_servers.synth_env.engine.rules import RulesEngine
    from resources_servers.synth_env.engine.state import StateEngine
    from resources_servers.synth_env.package_loader import load_package_zip
    from resources_servers.synth_env.profiles.office import OfficeProfile

    return {
        "load_package_zip": load_package_zip,
        "EventScheduler": EventScheduler,
        "NpcEngine": NpcEngine,
        "RulesEngine": RulesEngine,
        "StateEngine": StateEngine,
        "OfficeProfile": OfficeProfile,
    }


class _ExampleSession:
    def __init__(self, profile, state, rules, events, npc):
        from types import SimpleNamespace

        self.profile = profile
        self.state_engine = state
        self.rules = rules
        self.events = events
        self.npc = npc
        self.llm = None
        self.fs = profile.snapshot()
        self.baseline_fs = profile.snapshot()
        self.fidelity = SimpleNamespace(
            rule_strictness="block", npc_autonomy="strict"
        )
        self.budgets = SimpleNamespace(llm_sim_calls=None)
        self.llm_sim_calls = 0
        self.violations = []
        self.rule_events = []
        self.npc_events = []


def _verify_with_gym(asset_zip: Path, gym_root: Path) -> Dict[str, Any]:
    runtime = _load_gym_runtime(gym_root)
    package = runtime["load_package_zip"](asset_zip.read_bytes())
    results = []
    for task in package.tasks.values():
        state = runtime["StateEngine"](
            package.env_spec.state_schema, package.office.state_seed_rows
        )
        profile = runtime["OfficeProfile"](package.env_spec, package.fs_snapshot)
        session = _ExampleSession(
            profile,
            state,
            runtime["RulesEngine"](package.office.rules),
            runtime["EventScheduler"](package.office.event_model),
            runtime["NpcEngine"](package.office.npc_model),
        )
        parts = task.goal.split(chr(96))
        if len(parts) < 2 or "ticket " not in task.goal:
            raise ExampleError("Unable to parse the action from task goal: {}".format(task.goal))
        action = parts[1]
        ticket_id = task.goal.split("ticket ", 1)[1].split(" (", 1)[0]
        import asyncio

        asyncio.run(
            profile.execute(action, {"ticket_id": ticket_id}, session=session)
        )
        passed = profile.check_success(
            session.fs, session.state_engine, task.success_check.verifier_spec
        )
        results.append(
            {
                "taskId": task.task_id,
                "action": action,
                "ticketId": ticket_id,
                "verified": bool(passed),
            }
        )
    failed = [item for item in results if not item["verified"]]
    if failed:
        raise ExampleError("Gym runtime verification failed: {}".format(failed))
    return {
        "runtimeRoot": str(gym_root),
        "refId": package.manifest.ref_id,
        "version": package.manifest.version,
        "fingerprint": package.manifest.fingerprint,
        "taskCount": len(package.tasks),
        "tasks": results,
    }


def run_example(
    work_dir: Optional[str] = None,
    gym_root: Optional[str] = None,
    *,
    ocr_backend: str = "paddleocr",
    ocr_device: str = "cpu",
    ocr_lang: str = "en",
) -> Dict[str, Any]:
    _require_runtime(ocr_backend)
    root = _prepare_workspace(work_dir)
    inputs_dir = root / "inputs"
    inputs_dir.mkdir()

    docx_path = inputs_dir / "seed.docx"
    pdf_path = inputs_dir / "regulation.pdf"
    png_path = inputs_dir / "scan.png"
    _write_docx(docx_path)
    _write_pdf(pdf_path)
    _write_png(png_path)

    source_input_path = root / "source-input.json"
    seed_dir = root / "seed-import"
    _write_json(
        source_input_path,
        _source_input(
            docx_path,
            pdf_path,
            png_path,
            ocr_backend=ocr_backend,
            ocr_device=ocr_device,
            ocr_lang=ocr_lang,
        ),
    )
    source_result = _run(
        [
            sys.executable,
            "-m",
            "data_designer.pipelines.envsynth.cli.source_import",
            str(source_input_path),
            str(seed_dir),
        ]
    )
    source_summary = _last_json(source_result.stdout, label="env_source_worker")
    seed_manifest_path = Path(str(source_summary["seedManifestPath"]))
    seed_manifest = json.loads(seed_manifest_path.read_text(encoding="utf-8"))
    if seed_manifest.get("curatorSchema") != "parsed-doc-v2":
        raise ExampleError("The example did not produce a parsed-doc-v2 seed bundle")
    media_types = sorted(
        {
            str(item.get("mediaType"))
            for item in seed_manifest.get("files") or []
            if item.get("mediaType")
        }
    )
    if media_types != ["docx", "pdf", "png"]:
        raise ExampleError(
            "Curator did not parse the expected docx/pdf/png inputs: {}".format(media_types)
        )
    parsed_rows = [
        json.loads(line)
        for line in (seed_dir / "parsed" / "documents.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    ocr_rows = [
        row
        for row in parsed_rows
        if ((row.get("parsed_doc") or {}).get("provenance") or {}).get("operation")
        == "image_ocr"
    ]
    if len(ocr_rows) != 1:
        raise ExampleError("The example should produce one image OCR C-IR-2 row")
    ocr_provenance = ocr_rows[0]["parsed_doc"]["provenance"]
    if ocr_provenance.get("device") != ocr_device:
        raise ExampleError(
            "OCR device mismatch: expected={} actual={}".format(
                ocr_device, ocr_provenance.get("device")
            )
        )

    store_path = root / "artifact-store"
    compile_input_path = root / "compile-input.json"
    snapshot_path = root / "snapshot.json"
    _write_json(
        compile_input_path,
        _compile_input(
            asset_root=root / "assets",
            store_path=store_path,
            seed_manifest_path=seed_manifest_path,
            seed_manifest_sha256=str(source_summary["seedManifestSha256"]),
        ),
    )
    _run(
        [
            sys.executable,
            "-m",
            "data_designer.pipelines.envsynth.cli.synthesis_worker",
            str(compile_input_path),
            str(snapshot_path),
        ]
    )
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    store_check = _verify_store(store_path, snapshot)

    target_gym = _discover_gym_root(gym_root)
    publish_gym = target_gym or (root / "gym")
    if target_gym is None:
        (publish_gym / "overlay").mkdir(parents=True)
    publish_payload = {
        "storePath": str(store_path),
        "refId": REF_ID,
        "version": int(snapshot["envRelease"]["version"]),
        "name": BENCHMARK_NAME,
    }
    publish_result = json.loads(
        _run(
            [
                sys.executable,
                str(PUBLISH_SCRIPT),
                "--gym-root",
                str(publish_gym),
            ],
            stdin=json.dumps(publish_payload, ensure_ascii=False),
        ).stdout
    )
    benchmark = str(publish_result["benchmark"])
    if publish_result.get("status") != "published":
        raise ExampleError("Gym publish status should be published")
    if publish_result.get("runnable") is not True:
        raise ExampleError("Gym publish result should have runnable=true")
    asset_zip = publish_gym / "overlay" / publish_result["assets"]["packageZip"]
    if not asset_zip.is_file():
        raise ExampleError("Gym publish artifact zip does not exist: {}".format(asset_zip))

    prepare_path = publish_gym / "overlay" / publish_result["assets"]["prepareScript"]
    _run([sys.executable, str(prepare_path)])
    data_path = publish_gym / "overlay" / publish_result["assets"]["generatedData"]
    if not data_path.is_file():
        raise ExampleError("prepare.py did not generate data: {}".format(data_path))

    runtime = _verify_with_gym(asset_zip, target_gym) if target_gym else None

    summary = {
        "schemaVersion": SUMMARY_SCHEMA,
        "status": "ok",
        "workDir": str(root),
        "curator": {
            "schema": seed_manifest.get("curatorSchema"),
            "mediaTypes": media_types,
            "ocr": {
                "requestedBackend": ocr_backend,
                "backend": ocr_provenance.get("backend"),
                "device": ocr_provenance.get("device"),
                "modelId": ocr_provenance.get("model_id"),
            },
            "seedManifestPath": str(seed_manifest_path),
            "seedManifestSha256": source_summary["seedManifestSha256"],
        },
        "dataDesigner": {
            "storePath": str(store_path),
            "snapshotPath": str(snapshot_path),
            "release": snapshot["envRelease"],
            "taskset": snapshot["taskset"],
            "storeCheck": store_check,
        },
        "gym": {
            "root": str(publish_gym),
            "runtimeRoot": str(target_gym) if target_gym else None,
            "runtimeVerified": runtime is not None,
            "benchmark": benchmark,
            "status": publish_result.get("status"),
            "runnable": publish_result.get("runnable"),
            "assetZip": str(asset_zip),
            "prepareScript": str(prepare_path),
            "generatedData": str(data_path),
            "runtime": runtime,
        },
    }
    _write_json(root / SUMMARY_NAME, summary)
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the formal Curator -> DataForge -> Gym example"
    )
    parser.add_argument(
        "--work-dir",
        help="New directory for generated inputs and outputs; defaults to a unique temp dir",
    )
    parser.add_argument(
        "--gym-root",
        help=(
            "Gym checkout containing overlay/resources_servers/synth_env. "
            "Defaults to DATA_FORGE_GYM or a sibling Gym checkout."
        ),
    )
    parser.add_argument(
        "--ocr-backend",
        choices=["paddleocr", "mindocr", "auto"],
        default="paddleocr",
        help="CPU OCR backend used by the formal example",
    )
    parser.add_argument(
        "--ocr-device",
        choices=["cpu", "npu", "cuda"],
        default="cpu",
        help="OCR device; CPU is sufficient for the formal example",
    )
    parser.add_argument(
        "--ocr-lang",
        choices=["en", "zh", "mixed"],
        default="en",
        help="OCR language for the generated English scan",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else list(argv))
    try:
        summary = run_example(
            args.work_dir,
            args.gym_root,
            ocr_backend=args.ocr_backend,
            ocr_device=args.ocr_device,
            ocr_lang=args.ocr_lang,
        )
    except ExampleError as error:
        print("[envsynth-example] failed: {}".format(error), file=sys.stderr)
        return 1
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    print("\n[envsynth-example] Curator -> DataForge -> Gym: OK")
    if summary["gym"]["runtimeVerified"]:
        print(
            "[envsynth-example] Gym loaded the published package and verified {} task(s).".format(
                summary["gym"]["runtime"]["taskCount"]
            )
        )
    else:
        print(
            "[envsynth-example] Gym runtime not discovered; package/publish contract verified."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
