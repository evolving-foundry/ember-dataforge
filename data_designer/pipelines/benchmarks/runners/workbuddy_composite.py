"""wb_composite native runner — deterministic scoring phase (design §8, M2).

File protocol matches ``native.py:run_case`` (argv ``<request.json>
<result.json>``; request carries ``taskId/caseManifestPath/candidateRoot/
phase/publicItem``). The runner is the trusted verifier process pinned by
``runtime["verifier"]``.

Scoring pipeline per phase:

1. sandbox: the pinned runtime root (python + grading deps, bootstrapped once
   under ``WB_COMPOSITE_SANDBOX_BASE``, default ``/var/tmp``) is hardlink-farmed
   into a fresh chroot root; the pinned workspace (``env_pin``) unpacks to
   ``/workspace`` and the frozen ``tests/`` tree (``grading_source`` member
   pin from the seed archive) lands at ``/tests`` — the absolute paths the
   real bench grades under, byte-identical semantics without root (D4:
   python deps-level sandbox; grading code is trusted-side frozen).
2. candidate deliverables named by ``payload.artifacts`` overlay from
   ``candidateRoot`` into ``/workspace`` (D6 red line: only declared
   deliverables are judged).
3. the pinned grading invocation (mirroring ``tests/verifier.toml``) runs
   pytest on ``/tests/grading`` inside the chroot; ``eval_result.json`` /
   junit XML are parsed into the deterministic rule component with the
   ``score_core`` pass-rate semantics.
4. M4 evaluation phase: when a judge backend is configured (``WB_JUDGE_*``
   env), the frozen harness sidecar ``extract_declared_artifacts.py`` runs
   inside the sandbox (judge evidence is harness-produced, D6), rubrics are
   binary-judged with bounded concurrency and a four-tuple cache
   (``workbuddy_judge``), and ``score_merge`` weighted_sum fuses rule and
   llm components — a judge component can never overturn a rule failure.
   ``judge_error`` degrades to the pure rule score with an ``llm_missing``
   flag; without a backend the phase stays ``"skipped_no_backend"`` (offline
   CI never touches the network). Reference/negative phases remain
   deterministic (``llm_judge_status="audit_exempt"``).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import time
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

VERSION = "wb-composite-verifier-m4"

# Bump whenever the bootstrap layout changes: the version is part of the
# runtime cache path, so old trees (e.g. ones whose /usr/bin/python3 pointed at
# a different minor) are rebuilt instead of silently reused.
RUNTIME_LAYOUT_VERSION = 2
RULE_PASS_THRESHOLD = 1.0
SANDBOX_BASE_ENV = "WB_COMPOSITE_SANDBOX_BASE"
DEFAULT_SANDBOX_BASE = "/var/tmp"
_DEFAULT_TIMEOUT = 1800.0

# This module is executed as a script (verifier_command), so relative imports cannot be used;
# absolute imports after sys.path bootstrap work under both execution modes.
from data_designer.pipelines.benchmarks.runners import (  # noqa: E402
    workbuddy_judge as wb_judge,
)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
    )


def _clean_relative(value: Any) -> str:
    relative = str(value or "").strip()
    parts = Path(relative).parts
    if (
        not relative
        or Path(relative).is_absolute()
        or ".." in parts
        or not all(part not in ("", ".", "..") for part in parts)
    ):
        raise ValueError(f"path must be a clean relative path: {relative!r}")
    return Path(relative).as_posix()


def _within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _dir_content_sha256(root: Path) -> str:
    """Directory pin: sha256 over the sorted (relpath, file sha256) list.

    Recipe matches ``seeds/wbbench_seed_adapter._dir_content_sha256`` — the
    runner re-derives it over the extracted grading tree instead of trusting
    the pin blindly.
    """
    entries = [
        {
            "path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(p for p in root.rglob("*") if p.is_file())
    ]
    payload = json.dumps(
        entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# ── runtime root (bootstrap once, hardlink-farm per eval) ───────────────────


def _sandbox_base() -> Path:
    base = Path(os.environ.get(SANDBOX_BASE_ENV) or DEFAULT_SANDBOX_BASE)
    base.mkdir(parents=True, exist_ok=True)
    return base


def _rmtree_forcer(func, path, _exc_info) -> None:  # pragma: no cover - fs edge
    try:
        os.chmod(path, 0o700)
        func(path)
    except OSError:
        pass


def _copy_tree_best_effort(src: Path, dst: Path) -> None:
    try:
        shutil.copytree(
            src,
            dst,
            symlinks=True,
            ignore_dangling_symlinks=True,
            copy_function=shutil.copy2,
            dirs_exist_ok=True,
        )
    except (OSError, shutil.Error):
        pass


def _link_tree(src: Path, dst: Path) -> None:
    """Hardlink farm for a tree we own (fs.protected_hardlinks allows this)."""
    dst.mkdir(parents=True, exist_ok=True)
    for path in sorted(src.rglob("*")):
        target = dst / path.relative_to(src)
        if path.is_symlink():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                target.symlink_to(os.readlink(path))
            except OSError:
                pass
        elif path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(path, target)
            except OSError:
                shutil.copy2(path, target)


def _user_site_root() -> Optional[Path]:
    try:
        import site

        base = Path(site.getuserbase()) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    except Exception:
        return None
    return base if base.is_dir() else None


# Pure-python grading stack the frozen pytest trees may import inside the
# sandbox (pytest + its runtime deps; pyyaml for judge.yaml sidecars).
_GRADING_DEP_MODULES = (
    "pytest", "_pytest", "iniconfig", "pluggy", "packaging", "pygments", "py", "yaml",
)


def _copy_grading_deps(site_packages: Path) -> None:
    """Copy the grading stack from the trusted env into the sandbox env."""
    source_dirs: set[Path] = set()
    for name in _GRADING_DEP_MODULES:
        spec = importlib.util.find_spec(name)
        if spec is None or not spec.origin:
            continue
        origin = Path(spec.origin)
        if origin.name == "__init__.py":
            _copy_tree_best_effort(origin.parent, site_packages / name)
            source_dirs.add(origin.parent)
        else:
            site_packages.mkdir(parents=True, exist_ok=True)
            shutil.copy2(origin, site_packages / origin.name)
            source_dirs.add(origin.parent)
    if not (site_packages / "pytest").exists():
        raise RuntimeError(
            "wb_composite sandbox bootstrap: pytest is not importable from "
            f"the trusted interpreter (looked in {site_packages})"
        )
    _copy_dist_metadata(site_packages, source_dirs)


def _copy_dist_metadata(site_packages: Path, source_dirs: set[Path]) -> None:
    """``importlib.metadata`` (pluggy.__version__ etc.) needs dist-info dirs."""
    try:
        from importlib import metadata

        mapping = metadata.packages_distributions()
    except (ImportError, OSError):
        return
    wanted = {
        distribution.lower()
        for module in _GRADING_DEP_MODULES
        for distribution in mapping.get(module, ())
    }
    if not wanted:
        return
    for source in source_dirs:
        for entry in source.glob("*.dist-info"):
            if entry.name.lower().split("-")[0] in wanted:
                _copy_tree_best_effort(entry, site_packages / entry.name)


def _python_key() -> str:
    return f"python{sys.version_info.major}.{sys.version_info.minor}"


def _runtime_interpreter() -> Path:
    """The real interpreter backing this process (venv/symlink resolved)."""
    candidate = Path(os.path.realpath(sys.executable))
    if not candidate.is_file():
        raise RuntimeError(
            "cannot locate the interpreter running the verifier: "
            f"{sys.executable}"
        )
    return candidate


def _runtime_marker(root: Path, key: str) -> Dict[str, Any]:
    """Return the cached marker when it describes this exact bootstrap."""
    try:
        data = json.loads((root / "marker.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    expected = {
        "python": key,
        "layout": RUNTIME_LAYOUT_VERSION,
        "interpreter": str(_runtime_interpreter()),
    }
    if any(data.get(name) != value for name, value in expected.items()):
        return {}
    # ``is_file`` follows symlinks, so a dangling ``env``/python alias also
    # invalidates the cache instead of surfacing as pytest exit 127.
    for tool in (key, "env"):
        if not (root / "usr" / "bin" / tool).is_file():
            return {}
    return data


def _remove_path(path: Path) -> None:
    """Remove a file/symlink/directory if present (bootstrap temp trees)."""
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def _pin_runtime_interpreter(
    bin_dir: Path, interpreter: Path, key: str
) -> None:
    """Pin ``python``/``python3`` to the verifier's own interpreter minor.

    Copying ``/usr/bin`` wholesale also copies the host ``python3`` alias,
    which may point at another minor than the stdlib/user-site trees built
    here (host 3.14 vs verifier 3.10). Inside the chroot that mismatch makes
    ``python3 -m pytest`` fail before collection.
    """
    target = bin_dir / key
    _remove_path(target)
    shutil.copy2(interpreter, target)
    target.chmod(0o755)
    for alias in ("python", "python3"):
        link = bin_dir / alias
        _remove_path(link)
        link.symlink_to(key)


def _materialize_binary(bin_dir: Path, name: str) -> None:
    """Replace a copied symlink with the host binary it resolves to.

    ``/usr/bin/env`` may point into a tree the runtime does not copy (e.g.
    ``../lib/cargo/bin/coreutils/env``); inside the chroot that dangling link
    makes every ``env VAR=...`` invocation exit 127.
    """
    dest = bin_dir / name
    if dest.is_file() and not dest.is_symlink():
        return
    host = Path("/usr/bin") / name
    if not host.exists():
        return
    target = Path(os.path.realpath(host))
    if not target.is_file():
        return
    _remove_path(dest)
    shutil.copy2(target, dest)
    dest.chmod(0o755)


def _copy_shared_libpython(tmp: Path) -> None:
    """Carry libpython when the verifier interpreter is a shared build."""
    if not sysconfig.get_config_var("Py_ENABLE_SHARED"):
        return
    libdir = sysconfig.get_config_var("LIBDIR")
    library = sysconfig.get_config_var("LDLIBRARY")
    if not libdir or not library:
        return
    source = Path(str(libdir)) / str(library)
    if not source.is_file():
        return
    dest = tmp / "usr" / "lib" / "x86_64-linux-gnu"
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest / source.name)


def _runtime_root() -> Path:
    """Bootstrap the trusted-side python runtime copy (idempotent).

    The sandbox interpreter is pinned to the verifier's own interpreter —
    copying ``/usr/bin`` wholesale leaves the host ``python3`` alias, which
    may resolve to a different minor than the stdlib/grading trees built
    here; inside the chroot that mismatch makes ``python3 -m pytest`` fail
    before collection. The pure-python grading stack is copied from the
    trusted env so the sandbox works even when the deps live in a venv
    without a user-site mirror.
    """
    key = _python_key()
    root = _sandbox_base() / f"wb-runtime-{key}-v{RUNTIME_LAYOUT_VERSION}"
    if _runtime_marker(root, key):
        return root

    interpreter = _runtime_interpreter()
    stdlib = Path(sysconfig.get_path("stdlib") or "").resolve()
    if not (stdlib / "os.py").is_file():
        raise RuntimeError(
            f"stdlib for the verifier interpreter not found: {stdlib}"
        )

    if root.exists():
        shutil.rmtree(root, onerror=_rmtree_forcer)

    tmp = root.parent / f".{root.name}.build-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    (tmp / "usr" / "lib").mkdir(parents=True, exist_ok=True)
    (tmp / "etc").mkdir(parents=True, exist_ok=True)
    _copy_tree_best_effort(Path("/usr/bin"), tmp / "usr" / "bin")
    for stray in (tmp / "usr" / "bin").glob("python*"):
        _remove_path(stray)
    _pin_runtime_interpreter(tmp / "usr" / "bin", interpreter, key)
    _materialize_binary(tmp / "usr" / "bin", "env")
    _copy_tree_best_effort(stdlib, tmp / "usr" / "lib" / key)
    _copy_grading_deps(tmp / "usr" / "lib" / key / "site-packages")
    _copy_tree_best_effort(
        Path("/usr/lib/x86_64-linux-gnu"),
        tmp / "usr" / "lib" / "x86_64-linux-gnu",
    )
    _copy_tree_best_effort(Path("/usr/local/lib"), tmp / "usr" / "local" / "lib")
    _copy_shared_libpython(tmp)
    user_site = _user_site_root()
    if user_site is not None:
        _link_tree(user_site.parent, tmp / "home" / "sandbox" / ".local" / "lib")
    for name in (
        "passwd", "group", "hosts", "resolv.conf", "ld.so.cache",
        "ld.so.conf", "nsswitch.conf", "localtime",
    ):
        source = Path("/etc") / name
        if source.exists():
            try:
                shutil.copy2(source, tmp / "etc" / name)
            except OSError:
                pass
    ld_conf_d = Path("/etc/ld.so.conf.d")
    if ld_conf_d.is_dir():
        _copy_tree_best_effort(ld_conf_d, tmp / "etc" / "ld.so.conf.d")
    (tmp / "marker.json").write_text(
        json.dumps(
            {
                "python": key,
                "layout": RUNTIME_LAYOUT_VERSION,
                "version": sys.version.split()[0],
                "interpreter": str(interpreter),
                "stdlib": str(stdlib),
            }
        ),
        encoding="utf-8",
    )
    tmp.rename(root)
    return root


def _build_sandbox() -> Path:
    """Fresh chroot root with the pinned absolute-path layout."""
    base = _sandbox_base()
    runtime = _runtime_root()
    root = Path(tempfile.mkdtemp(prefix="wb-composite-eval-", dir=str(base)))
    (root / ".runtime").mkdir()
    subprocess.run(
        ["cp", "-al", str(runtime / "usr"), str(root / ".runtime" / "usr")],
        check=True,
    )
    if (runtime / "home" / "sandbox").is_dir():
        subprocess.run(
            ["cp", "-al", str(runtime / "home"), str(root / ".runtime" / "home")],
            check=True,
        )
    shutil.copytree(runtime / "etc", root / ".runtime" / "etc", dirs_exist_ok=True)
    os.symlink(".runtime/usr", root / "usr")
    os.symlink(".runtime/etc", root / "etc")
    if (root / ".runtime/home").is_dir():
        os.symlink(".runtime/home", root / "home")
    (root / "lib64").mkdir(exist_ok=True)
    loader = root / "usr" / "lib" / "x86_64-linux-gnu" / "ld-linux-x86-64.so.2"
    if loader.exists():
        os.symlink(
            "/usr/lib/x86_64-linux-gnu/ld-linux-x86-64.so.2",
            root / "lib64" / "ld-linux-x86-64.so.2",
        )
    os.symlink("usr/bin", root / "bin")
    os.symlink("usr/lib", root / "lib")
    for name in ("workspace", "tests", "dev"):
        (root / name).mkdir(exist_ok=True)
    # No device nodes inside the userns chroot; pytest capture only opens
    # /dev/null (read mode) as its dup2 sink, so an empty regular file
    # satisfies it without mknod privileges.
    dev_null = root / "dev" / "null"
    dev_null.touch()
    os.chmod(dev_null, 0o666)
    (root / "logs" / "verifier").mkdir(parents=True, exist_ok=True)
    return root


def _teardown_sandbox(root: Path) -> None:
    shutil.rmtree(root, onerror=_rmtree_forcer)


# ── pinned material extraction ──────────────────────────────────────────────


def _safe_members(tar: tarfile.TarFile) -> List[tarfile.TarInfo]:
    members: List[tarfile.TarInfo] = []
    for member in tar.getmembers():
        relative = Path(member.name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"archive member escapes layout: {member.name}")
        if member.issym() or member.islnk():
            link = Path(member.linkname)
            if link.is_absolute() or ".." in link.parts:
                raise ValueError(f"archive link escapes layout: {member.name}")
        members.append(member)
    return members


def _extract_workspace(pin: Dict[str, Any], dest: Path) -> None:
    archive = Path(str(pin.get("path") or ""))
    sha = str(pin.get("sha256") or "")
    if not archive.is_absolute() or not archive.is_file():
        raise ValueError(f"pinned workspace missing: {archive}")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != sha:
        raise ValueError(f"pinned workspace SHA-256 drift: {archive}")
    with tarfile.open(archive, "r:gz") as tar:
        for member in _safe_members(tar):
            if not member.isfile():
                continue
            target = dest / Path(member.name)
            if not _within(target, dest):
                raise ValueError(f"workspace member escapes root: {member.name}")
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"workspace member unreadable: {member.name}")
            target.write_bytes(stream.read())


def _extract_grading(grading_source: Dict[str, Any], dest: Path) -> None:
    """Unpack ``<member_prefix>/`` from the pinned frozen seed archive."""
    archive = Path(str(grading_source.get("path") or ""))
    sha = str(grading_source.get("sha256") or "")
    prefix = _clean_relative(grading_source.get("member_prefix"))
    if not archive.is_absolute() or not archive.is_file():
        raise ValueError(f"pinned grading source missing: {archive}")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != sha:
        raise ValueError(f"pinned grading source SHA-256 drift: {archive}")
    full_prefix = f"{prefix}/"
    with tarfile.open(archive, "r:gz") as tar:
        members = _safe_members(tar)
        tops = {Path(m.name).parts[0] for m in members if m.name.strip()}
        if len(tops) != 1:
            raise ValueError("grading archive must have a single top-level root")
        archive_prefix = f"{tops.pop()}/{full_prefix}"
        selected = [
            m
            for m in members
            if m.name.startswith(archive_prefix) and m.isfile()
        ]
        if not selected:
            raise ValueError(
                f"grading members absent from pinned archive: {prefix}"
            )
        for member in selected:
            relative = Path(member.name).relative_to(archive_prefix)
            target = dest / relative
            if not _within(target, dest):
                raise ValueError(
                    f"grading member escapes tests root: {member.name}"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"grading member unreadable: {member.name}")
            target.write_bytes(stream.read())


def _overlay_artifacts(
    payload: Dict[str, Any], candidate_root: Path, workspace: Path
) -> List[Dict[str, Any]]:
    """Copy declared deliverables from candidateRoot into /workspace (D6)."""
    overlays: List[Dict[str, Any]] = []
    for artifact in payload.get("artifacts") or []:
        relative = _clean_relative(artifact.get("path"))
        source = candidate_root / relative
        if source.is_symlink() or not _within(source, candidate_root):
            raise FileNotFoundError(
                f"deliverable outside candidateRoot: {relative}"
            )
        if not source.is_file():
            if artifact.get("required"):
                raise FileNotFoundError(
                    f"required deliverable missing: {relative}"
                )
            continue
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        data = source.read_bytes()
        target.write_bytes(data)
        overlays.append(
            {
                "id": str(artifact.get("id") or relative),
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )
    return overlays


# ── grading invocation + rule parsing ───────────────────────────────────────


def _pytest_inner_command() -> str:
    # Env mirrors the frozen harness verifier.toml [env] exactly. WB_BENCH_
    # OUTPUT_PATH / CASE_DIR must stay UNSET: 50/50 graders read them but the
    # harness never provides them, so each grader's task-specific
    # DEFAULT_OUTPUT_PATH (file vs directory semantics) is authoritative.
    env = (
        "PYTHONPATH=/workspace "
        "WB_BENCH_GOLD_PATH=/tests/gold/gold_answer.json "
        "WB_BENCH_FIXTURES_DIR=/tests/gold/fixtures "
        "HOME=/home/sandbox "
        "PYTHONDONTWRITEBYTECODE=1 "
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    )
    return (
        f"cd /workspace && env {env} python3 -m pytest /tests/grading "
        "-p no:cacheprovider -v --tb=short "
        "--junitxml=/logs/verifier/results.xml "
        "> /logs/verifier/test_output.txt 2>&1; echo PYTEST_EXIT=$?"
    )


def _run_pytest(root: Path, timeout: float) -> Tuple[int, bool]:
    """Run the pinned grading invocation inside the userns chroot.

    Returns ``(pytest_exit, timed_out)``; exit -1 marks an infra failure.
    """
    inner = _pytest_inner_command()
    quoted_root = shlex.quote(str(root))
    script = f"chroot {quoted_root} /bin/bash -c {shlex.quote(inner)}"
    try:
        proc = subprocess.run(
            ["unshare", "-r", "-m", "bash", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            start_new_session=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "unshare unavailable: wb_composite sandbox requires "
            "Linux user namespaces"
        ) from error
    except subprocess.TimeoutExpired:
        return -1, True
    match = re.search(r"PYTEST_EXIT=(\d+)\s*$", proc.stdout or "")
    if match:
        return int(match.group(1)), False
    return -1, False


def _judge_sidecar_invocation(root: Path, timeout: float) -> None:
    """Run the frozen harness judge-evidence sidecar inside the chroot (M4).

    ``extract_declared_artifacts.py`` extracts rubric-referenced declared
    artifacts from /workspace into /logs/verifier/artifact_text (xlsx → text,
    unsafe-evidence interception) and writes ``artifact_manifest.json``. A
    nonzero exit (``needs_review``) is not fatal — missing/broken evidence
    surfaces explicitly through the judge loader.
    """
    inner = (
        "cd /workspace && env HOME=/home/sandbox PYTHONDONTWRITEBYTECODE=1 "
        "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "
        "python3 /tests/grading/judge/extract_declared_artifacts.py "
        "--judge-yaml /tests/judge.yaml --workspace /workspace "
        "--logs /logs/verifier >/logs/verifier/sidecar_output.txt 2>&1; "
        "echo SIDECAR_EXIT=$?"
    )
    script = f"chroot {shlex.quote(str(root))} /bin/bash -c {shlex.quote(inner)}"
    try:
        subprocess.run(
            ["unshare", "-r", "-m", "bash", "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
            start_new_session=True,
        )
    except subprocess.TimeoutExpired:
        return  # manifest will be absent → judge loader fails explicitly
    except FileNotFoundError:
        return  # unshare unavailable → judge loader fails explicitly


def _junit_counts(results_xml: Path) -> Dict[str, int]:
    counts = {
        "tests_collected": 0, "tests_failed": 0, "tests_error": 0,
        "tests_skipped": 0, "tests_passed": 0,
    }
    if not results_xml.is_file():
        return counts
    try:
        tree_root = ET.parse(results_xml).getroot()
        suites = (
            tree_root.findall("testsuite")
            if tree_root.tag == "testsuites"
            else [tree_root]
        )
        collected = sum(int(s.get("tests", 0)) for s in suites)
        failed = sum(int(s.get("failures", 0)) for s in suites)
        errors = sum(int(s.get("errors", 0)) for s in suites)
        skipped = sum(int(s.get("skipped", 0)) for s in suites)
        counts = {
            "tests_collected": collected,
            "tests_failed": failed,
            "tests_error": errors,
            "tests_skipped": skipped,
            "tests_passed": max(0, collected - failed - errors - skipped),
        }
    except ET.ParseError:
        pass
    return counts


def _rule_component(root: Path, pytest_exit: int) -> Dict[str, Any]:
    """Deterministic rule component (score_core pass-rate semantics)."""
    logs = root / "logs" / "verifier"
    output_text = ""
    raw_output = logs / "test_output.txt"
    if raw_output.is_file():
        output_text = raw_output.read_text(encoding="utf-8", errors="replace")
    markers = (
        "ERROR: found no collectors",
        "ImportError while loading conftest",
        "fixture '",
    )
    junit = _junit_counts(logs / "results.xml")
    junit_total = max(0, junit["tests_collected"] - junit["tests_skipped"])

    eval_result: Dict[str, Any] = {}
    eval_path = logs / "eval_result.json"
    if eval_path.is_file():
        try:
            loaded = _load_json(eval_path)
            if isinstance(loaded, dict):
                eval_result = loaded
        except ValueError:
            eval_result = {}
    checks = (
        eval_result.get("checks")
        if isinstance(eval_result.get("checks"), list)
        else []
    )
    if checks:
        total = int(eval_result.get("total_count") or len(checks))
        passed = int(
            eval_result.get("passed_count")
            or sum(1 for check in checks if check.get("passed") is True)
        )
        rate: Optional[float] = None
        if eval_result.get("pass_rate") is not None:
            try:
                rate = round(float(eval_result["pass_rate"]), 4)
            except (TypeError, ValueError):
                rate = None
        if rate is None:
            rate = round(passed / total, 4) if total else 0.0
        source = "eval_result"
    else:
        passed, total = junit["tests_passed"], junit_total
        rate = round(passed / total, 4) if total else 0.0
        source = "junit"

    if pytest_exit not in (0, 1) or total <= 0 or any(
        marker in output_text for marker in markers
    ):
        status = "build_error"
    elif rate >= 1.0:
        status = "full_pass"
    elif rate > 0:
        status = "partial_pass"
    else:
        status = "no_pass"

    failed_checks = [
        {
            "name": str(check.get("name", "unnamed_check")),
            "detail": str(check.get("detail", "")),
        }
        for check in checks
        if check.get("passed") is not True
    ][:20]
    return {
        "rule_pass_rate": rate,
        "rule_passed": passed,
        "rule_total": total,
        "rule_source": source,
        "rule_status": status,
        "pytest_exit": pytest_exit,
        "failed_checks": failed_checks,
    }


# ── request scoring ─────────────────────────────────────────────────────────


def _invalid_candidate(
    task_id: str, started: float, reason: str
) -> Dict[str, Any]:
    return {
        "task_id": task_id,
        "status": "invalid_candidate",
        "passed": False,
        "score": 0.0,
        "native_metrics": {"artifact_bytes": 0.0},
        "rule_diagnostics": {"invalid_reason": reason},
        "verifier_version": VERSION,
        "evidence_refs": [],
        "runtime_seconds": time.monotonic() - started,
    }


def score_request(request: Dict[str, Any]) -> Dict[str, Any]:
    started = time.monotonic()
    manifest = _load_json(Path(request["caseManifestPath"]))
    if manifest.get("task_id") != request.get("taskId"):
        raise ValueError("workbuddy task ID disagrees with trusted manifest")
    if manifest.get("family") != "workbuddy":
        raise ValueError("trusted manifest is not a workbuddy case")
    spec = manifest.get("private", {}).get("verifier_spec")
    if not isinstance(spec, dict) or spec.get("kind") != "wb_composite":
        raise ValueError("trusted manifest lacks a wb_composite verifier_spec")
    payload = spec.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("verifier_spec payload missing")
    grading_source = payload.get("grading_source")
    if not isinstance(grading_source, dict):
        raise ValueError("wb_composite payload needs a grading_source pin")
    env_pin = payload.get("env_pin")
    if not isinstance(env_pin, dict):
        raise ValueError("wb_composite payload needs a workspace env_pin")
    phase = request.get("phase")
    if phase not in ("reference", "negative", "evaluation"):
        raise ValueError(f"unknown phase: {phase!r}")
    task_id = request["taskId"]
    candidate_root = Path(request["candidateRoot"]).resolve(strict=True)

    root = _build_sandbox()
    try:
        _extract_workspace(env_pin, root / "workspace")
        _extract_grading(grading_source, root / "tests")
        grading_dir = root / "tests" / "grading"
        expected_grading_sha = payload.get("grading_dir_sha256")
        if (
            not grading_dir.is_dir()
            or _dir_content_sha256(grading_dir) != expected_grading_sha
        ):
            raise ValueError("extracted grading dir fails its content pin")
        gold_path = root / "tests" / "gold" / "gold_answer.json"
        if _load_json(gold_path) != payload.get("gold"):
            raise ValueError(
                "extracted gold_answer.json disagrees with the frozen gold"
            )
        try:
            overlays = _overlay_artifacts(
                payload, candidate_root, root / "workspace"
            )
        except FileNotFoundError as error:
            if phase in ("reference", "negative"):
                raise ValueError(
                    f"audit deliverable invalid for {task_id}: {error}"
                )
            return _invalid_candidate(task_id, started, str(error))
        timeout_value = payload.get("timeout_sec")
        timeout = float(timeout_value) if timeout_value else _DEFAULT_TIMEOUT
        pytest_exit, timed_out = _run_pytest(root, timeout)
        rules = _rule_component(root, pytest_exit)
        logs = root / "logs" / "verifier"
        if (
            not (logs / "results.xml").is_file()
            and not (logs / "eval_result.json").is_file()
        ):
            detail = ""
            output = logs / "test_output.txt"
            if output.is_file():
                text = output.read_text(encoding="utf-8", errors="replace")
                detail = f"\n--- sandbox output tail ---\n{text[-2000:]}"
            raise RuntimeError(
                "grading produced no results inside the sandbox "
                f"(pytest_exit={pytest_exit}, timed_out={timed_out})"
                f"{detail}"
            )
        judge_cfg = (
            wb_judge.backend_config() if phase == "evaluation" else None
        )
        judge_result: Optional[Dict[str, Any]] = None
        if judge_cfg is not None:
            _judge_sidecar_invocation(root, timeout)
            try:
                evidence_by_ref, sha_by_ref = wb_judge.load_judge_evidence(
                    str(root / "logs" / "verifier"),
                    payload.get("artifacts") or [],
                )
                judge_result = wb_judge.judge_task_rubrics(
                    payload.get("rubrics") or [],
                    evidence_by_ref,
                    sha_by_ref,
                    judge_cfg,
                    cache_dir=os.environ.get("WB_JUDGE_CACHE_DIR") or None,
                )
            except Exception as error:  # noqa: BLE001 — explicit judge_error degradation
                judge_result = {
                    "status": "judge_error",
                    "llm_score": None,
                    "errors": [repr(error)],
                }
    finally:
        _teardown_sandbox(root)

    rate = float(rules["rule_pass_rate"])
    if not 0.0 <= rate <= 1.0:
        raise ValueError("rule pass rate outside [0,1]")
    score_merge = payload.get("score_merge")
    judge_diagnostics: Dict[str, Any] = {}
    if phase in ("reference", "negative"):
        # Audit arms keep deterministic semantics (M2 acceptance baseline); the judge component does not participate.
        llm_status = "audit_exempt"
        passed = rate >= RULE_PASS_THRESHOLD
        score = rate
    else:
        if judge_result is None:
            llm_status = "skipped_no_backend"
        else:
            llm_status = (
                "judge_ok"
                if judge_result.get("status") == "judge_ok"
                else "judge_error"
            )
        merged = wb_judge.merge_scores(
            rate,
            judge_result,
            score_merge if isinstance(score_merge, dict) else {},
            rules,
        )
        passed = merged["passed"]
        score = merged["score"]
        judge_diagnostics = {
            "judge_status": (
                judge_result.get("status") if judge_result else None
            ),
            "judge_errors": (judge_result or {}).get("errors") or [],
            "merge_source": merged["merge_source"],
            "llm_missing": merged["llm_missing"],
            "critical_failure": merged["critical_failure"],
            "llm_score": merged["llm_score"],
            "pass_threshold": merged["pass_threshold"],
        }
        if judge_result:
            judge_diagnostics.update(
                {
                    "judge_fingerprint": judge_result.get("fingerprint"),
                    "judge_call_count": judge_result.get("call_count", 0),
                    "judge_cached_count": judge_result.get("cached_count", 0),
                    "judge_total_latency_seconds": judge_result.get(
                        "total_latency_seconds"
                    ),
                    "judge_prompt_tokens": judge_result.get(
                        "prompt_tokens", 0
                    ),
                    "judge_completion_tokens": judge_result.get(
                        "completion_tokens", 0
                    ),
                    "judge_rubric_results": (
                        judge_result.get("rubric_results") or []
                    ),
                }
            )
    response = {
        "task_id": task_id,
        "status": "completed",
        "passed": passed,
        "score": score,
        "llm_judge_status": llm_status,
        # validate_native_result requires numeric-only native_metrics; the
        # string/list diagnostics travel in rule_diagnostics (evidence only).
        "native_metrics": {
            "artifact_bytes": float(sum(o["bytes"] for o in overlays)),
            "rule_pass_rate": rules["rule_pass_rate"],
            "rule_passed": float(rules["rule_passed"]),
            "rule_total": float(rules["rule_total"]),
            "pytest_exit": float(pytest_exit),
        },
        "rule_diagnostics": {
            "rule_source": rules["rule_source"],
            "rule_status": rules["rule_status"],
            "failed_checks": rules["failed_checks"],
            "timed_out": timed_out,
            "judge": judge_diagnostics,
        },
        "verifier_version": VERSION,
        "evidence_refs": [
            f"deliverable-sha256:{o['path']}:{o['sha256']}" for o in overlays
        ],
        "runtime_seconds": time.monotonic() - started,
    }
    if "attempt" in request:
        response.update(
            attempt_index=request["attempt"]["index"],
            seed=request["attempt"]["seed"],
        )
    return response


def main() -> None:
    if len(sys.argv) != 3:
        raise ValueError(
            "usage: workbuddy_composite.py <request.json> <result.json>"
        )
    request = _load_json(Path(sys.argv[1]))
    result = score_request(request)
    Path(sys.argv[2]).write_text(
        json.dumps(result, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
