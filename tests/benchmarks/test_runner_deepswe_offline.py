"""deepswe offline runner tests (deepswe follow-up M1): full branch coverage for offline git apply scoring.

Aligned with the deepswe scoring re-expression follow-up doc (v0.1) M1 acceptance: the runner returns passed=True/score=1.0 on the gold diff,
completed/passed=False/0.0 on bad patches/wrong content/out-of-bounds patches, invalid_candidate for missing/oversized deliverables,
and an explicit failure when the audit stage is missing; the binary diff (the official collector uses ``git diff --binary``) is validated with a real
git-generated patch containing a ``GIT binary patch`` section, proving both apply and SHA-256 comparison. All results pass
``validate_native_result`` (enforced inside ``run_case``). Fully offline, depends only on git.
"""

import difflib
import hashlib
import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest

from data_designer.pipelines.benchmarks.contracts import NativeFailure
from data_designer.pipelines.benchmarks.native import run_case
from data_designer.pipelines.benchmarks.runners.deepswe_offline import (
    MAX_PATCH_BYTES,
    VERSION,
)

RUNNER_PATH = Path(__file__).resolve().parents[2] / (
    "data_designer/pipelines/benchmarks/runners/deepswe_offline.py"
)

TASK_ID = "deepswe:fixture-issue-1:v1"

BINARY_BASE = bytes(range(256)) + b"\x00binary base tail"
BINARY_NEW = b"\x89PNG\r\n\x1a\n" + bytes(reversed(range(256)))
BASE_FILES = {
    "src/app.py": "value = 1\n",
    "docs/readme.md": "hello\n",
    "to_delete.txt": "obsolete\n",
    "assets/logo.bin": BINARY_BASE,
}
GOLD_NEW = {
    "src/app.py": "value = 2\n",
    "assets/logo.bin": BINARY_NEW,
}
GOLD_DELETED = ["to_delete.txt"]


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=fixture@invalid",
         "-c", "user.name=fixture", *args],
        check=True, capture_output=True,
    )


def _diff_binary(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "diff", "--binary", "HEAD"],
        check=True, capture_output=True,
    )
    return completed.stdout.decode("utf-8")


def _mini_repo(files: dict) -> Path:
    repo = Path(tempfile.mkdtemp(prefix="deepswe-runner-repo-"))
    for relative, content in files.items():
        target = repo / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, str):
            target.write_text(content, encoding="utf-8")
        else:
            target.write_bytes(content)
    _git(repo, "init", "-q")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    return repo


def _gold_patch(binary_only: bool = False) -> str:
    repo = _mini_repo(BASE_FILES)
    changes = {"assets/logo.bin": BINARY_NEW} if binary_only else GOLD_NEW
    for relative, content in changes.items():
        target = repo / relative
        if isinstance(content, str):
            target.write_text(content, encoding="utf-8")
        else:
            target.write_bytes(content)
    patch = _diff_binary(repo)
    if not binary_only:
        patch += (
            "diff --git a/to_delete.txt b/to_delete.txt\n"
            "deleted file mode 100644\n"
            "--- a/to_delete.txt\n"
            "+++ /dev/null\n"
            "@@ -1 +0,0 @@\n"
            "-obsolete\n"
        )
    return patch


def _corrupted_patch() -> str:
    # the text hunk has one context-free removal line; corrupt it so git apply
    # must reject the patch outright
    return _gold_patch().replace("-value = 1\n", "-value = 1 tampered\n")


def _corrupted_binary_patch() -> str:
    # bump the literal payload length so the GIT binary patch fails to decode
    patch = _gold_patch(binary_only=True)
    return patch.replace("literal ", "literal 9", 1)


def _unified(old: str, new: str, relative: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{relative}",
            tofile=f"b/{relative}",
        )
    )


def _expected_files(gold_new: dict | None = None) -> dict:
    return {
        rel: _sha256_bytes(
            content if isinstance(content, bytes) else content.encode("utf-8")
        )
        for rel, content in (gold_new if gold_new is not None else GOLD_NEW).items()
    }


def _build_spec(base_zip_sha: str) -> dict:
    return {
        "kind": "deepswe_offline_patch",
        "generator_ref": "deepswe_offline_regression@fixture",
        "payload": {
            "deliverable": "patch.diff",
            "base_snapshot": {"path": "base.zip", "sha256": base_zip_sha},
            "expected_files": _expected_files(),
            "deleted_paths": list(GOLD_DELETED),
            "timeout_s": 30.0,
        },
    }


def _write_base_zip(tmp_path: Path) -> str:
    trusted = tmp_path / "trusted"
    trusted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(trusted / "base.zip", "w") as archive:
        for relative, content in BASE_FILES.items():
            archive.writestr(relative, content)
    return _sha256_bytes((trusted / "base.zip").read_bytes())


def _write_manifest(tmp_path: Path, spec: dict) -> str:
    manifest = {
        "schema_version": "benchmark-case-v1",
        "task_id": TASK_ID,
        "family": "deepswe",
        "private": {"verifier_spec": spec},
    }
    manifest_path = tmp_path / "trusted" / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    return str(manifest_path)


def _default_fixture(tmp_path: Path) -> str:
    return _write_manifest(tmp_path, _build_spec(_write_base_zip(tmp_path)))


def _runner_command():
    return [sys.executable, str(RUNNER_PATH)]


def _write_patch(root: Path, patch_text: str, relative: str = "patch.diff") -> None:
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(patch_text, encoding="utf-8")


def test_reference_phase_passes_on_gold_patch(tmp_path):
    manifest_path = _default_fixture(tmp_path)
    root = tmp_path / "reference"
    _write_patch(root, _gold_patch())
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="reference", timeout=30, public_item={},
    )
    assert result["status"] == "completed"
    assert result["passed"] is True
    assert result["score"] == 1.0
    assert result["verifier_version"] == VERSION
    assert result["native_metrics"]["compared_files"] == 3.0


def test_negative_phase_fails_on_corrupted_patch(tmp_path):
    manifest_path = _default_fixture(tmp_path)
    root = tmp_path / "negative"
    _write_patch(root, _corrupted_patch())
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="negative", timeout=30, public_item={},
    )
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_evaluation_acknowledges_attempt_seed(tmp_path):
    manifest_path = _default_fixture(tmp_path)
    root = tmp_path / "candidate"
    _write_patch(root, _gold_patch())
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="evaluation", timeout=30, public_item={},
        attempt={"index": 3, "seed": 777},
    )
    assert result["attempt_index"] == 3
    assert result["seed"] == 777
    assert result["passed"] is True


def test_binary_diff_gold_patch_passes(tmp_path):
    """M1 binary validation: a patch containing only a GIT binary patch section applies and hash comparison succeeds."""
    spec = _build_spec(_write_base_zip(tmp_path))
    spec["payload"]["expected_files"] = {
        "assets/logo.bin": _expected_files()["assets/logo.bin"],
    }
    spec["payload"]["deleted_paths"] = []
    manifest_path = _write_manifest(tmp_path, spec)
    root = tmp_path / "candidate"
    _write_patch(root, _gold_patch(binary_only=True))
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="evaluation", timeout=30, public_item={},
    )
    assert result["status"] == "completed"
    assert result["passed"] is True
    assert result["score"] == 1.0


def test_corrupted_binary_patch_scores_zero(tmp_path):
    spec = _build_spec(_write_base_zip(tmp_path))
    spec["payload"]["expected_files"] = {
        "assets/logo.bin": _expected_files()["assets/logo.bin"],
    }
    spec["payload"]["deleted_paths"] = []
    manifest_path = _write_manifest(tmp_path, spec)
    root = tmp_path / "candidate"
    _write_patch(root, _corrupted_binary_patch())
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="evaluation", timeout=30, public_item={},
    )
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_wrong_content_patch_applies_but_scores_zero(tmp_path):
    manifest_path = _default_fixture(tmp_path)
    root = tmp_path / "candidate"
    patch = _unified("value = 1\n", "value = 3\n", "src/app.py")
    _write_patch(root, patch)
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="evaluation", timeout=30, public_item={},
    )
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_out_of_scope_patch_scores_zero(tmp_path):
    manifest_path = _default_fixture(tmp_path)
    root = tmp_path / "candidate"
    _write_patch(root, _unified("hello\n", "changed\n", "docs/other.md"))
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="evaluation", timeout=30, public_item={},
    )
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_escape_path_patch_scores_zero(tmp_path):
    manifest_path = _default_fixture(tmp_path)
    root = tmp_path / "candidate"
    escape = (
        "--- a/../evil.txt\n"
        "+++ b/../evil.txt\n"
        "@@ -0,0 +1 @@\n"
        "+pwned\n"
    )
    _write_patch(root, escape)
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="evaluation", timeout=30, public_item={},
    )
    assert result["status"] == "completed"
    assert result["passed"] is False
    assert result["score"] == 0.0
    assert not (tmp_path / "evil.txt").exists()


def test_missing_deliverable_is_invalid_candidate_not_silent(tmp_path):
    manifest_path = _default_fixture(tmp_path)
    root = tmp_path / "candidate"
    root.mkdir()
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="evaluation", timeout=30, public_item={},
    )
    assert result["status"] == "invalid_candidate"
    assert result["passed"] is False
    assert result["score"] == 0.0


def test_oversized_patch_is_invalid_candidate(tmp_path):
    manifest_path = _default_fixture(tmp_path)
    root = tmp_path / "candidate"
    target = root / "patch.diff"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x" * (MAX_PATCH_BYTES + 1))
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="evaluation", timeout=30, public_item={},
    )
    assert result["status"] == "invalid_candidate"
    assert result["score"] == 0.0


def test_symlink_deliverable_is_invalid_candidate(tmp_path):
    manifest_path = _default_fixture(tmp_path)
    root = tmp_path / "candidate"
    outside = tmp_path / "outside.patch"
    outside.write_text(_gold_patch(), encoding="utf-8")
    target = root / "patch.diff"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(outside)
    result = run_case(
        _runner_command(), TASK_ID, manifest_path, str(root),
        phase="evaluation", timeout=30, public_item={},
    )
    assert result["status"] == "invalid_candidate"
    assert result["passed"] is False


def test_audit_phase_requires_deliverable(tmp_path):
    manifest_path = _default_fixture(tmp_path)
    root = tmp_path / "reference"
    root.mkdir()
    with pytest.raises(NativeFailure, match="native runner failed"):
        run_case(
            _runner_command(), TASK_ID, manifest_path, str(root),
            phase="reference", timeout=30, public_item={},
        )


def test_out_of_root_deliverable_path_is_rejected(tmp_path):
    spec = _build_spec(_write_base_zip(tmp_path))
    spec["payload"]["deliverable"] = "../escape.patch"
    manifest_path = _write_manifest(tmp_path, spec)
    root = tmp_path / "candidate"
    root.mkdir()
    with pytest.raises(NativeFailure, match="clean relative path"):
        run_case(
            _runner_command(), TASK_ID, manifest_path, str(root),
            phase="evaluation", timeout=30, public_item={},
        )


def test_base_snapshot_hash_mismatch_fails_loudly(tmp_path):
    _write_base_zip(tmp_path)
    spec = _build_spec("0" * 64)
    manifest_path = _write_manifest(tmp_path, spec)
    root = tmp_path / "candidate"
    _write_patch(root, _gold_patch())
    with pytest.raises(NativeFailure, match="sha256 mismatch"):
        run_case(
            _runner_command(), TASK_ID, manifest_path, str(root),
            phase="evaluation", timeout=30, public_item={},
        )


def test_unsupported_verifier_kind_is_rejected(tmp_path):
    spec = _build_spec(_write_base_zip(tmp_path))
    spec["kind"] = "swe_offline_patch"
    manifest_path = _write_manifest(tmp_path, spec)
    root = tmp_path / "candidate"
    _write_patch(root, _gold_patch())
    with pytest.raises(NativeFailure, match="unsupported verifier kind"):
        run_case(
            _runner_command(), TASK_ID, manifest_path, str(root),
            phase="evaluation", timeout=30, public_item={},
        )


def test_family_mismatch_is_rejected(tmp_path):
    spec = _build_spec(_write_base_zip(tmp_path))
    manifest_path = _write_manifest(tmp_path, spec)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest["family"] = "swe"
    Path(manifest_path).write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    root = tmp_path / "candidate"
    _write_patch(root, _gold_patch())
    with pytest.raises(NativeFailure, match="not a deepswe case"):
        run_case(
            _runner_command(), TASK_ID, manifest_path, str(root),
            phase="evaluation", timeout=30, public_item={},
        )
