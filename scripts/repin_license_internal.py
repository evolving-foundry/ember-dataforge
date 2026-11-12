"""In-place license re-pin for swe/osworld/gaia source manifests.

Lab-side operational script (run with the designer venv that has
``data_designer`` importable). Flips ``license.derive_allowed`` to
``true`` with explicit decision metadata (per the 2026-10-08 decision 1:
internal evaluation use, license archived not gated), keeps a byte-exact
``.bak-<date>`` copy of the original, re-verifies the manifest against its
pinned upstream snapshot, and prints the new SHA-256 for the ledger.

Usage:
    python repin_license_internal.py <source.json> [<source.json> ...]

Only families swe / osworld / gaia are accepted. The script is idempotent:
manifests already carrying ``derive_allowed=true`` are skipped unchanged.
"""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from data_designer.pipelines.benchmarks.families.swe import (
    verify_swe_source_manifest,
)
from data_designer.pipelines.benchmarks.families.osworld import (
    verify_osworld_source_manifest,
)
from data_designer.pipelines.benchmarks.families.gaia import (
    verify_gaia_source_manifest,
)

VERIFIERS = {
    "swe": verify_swe_source_manifest,
    "osworld": verify_osworld_source_manifest,
    "gaia": verify_gaia_source_manifest,
}
DECIDED_BY = "chadwweng"


def repin(path: Path) -> str:
    raw = path.read_bytes()
    source = json.loads(raw.decode("utf-8"))
    family = source.get("family")
    if family not in VERIFIERS:
        raise SystemExit(f"{path}: family {family!r} not in {sorted(VERIFIERS)}")
    license_record = source.get("license")
    if not isinstance(license_record, dict):
        raise SystemExit(f"{path}: missing license record")
    if license_record.get("derive_allowed") is True:
        print(f"[skip] {path}: derive_allowed already true")
        return hashlib.sha256(raw).hexdigest()

    verify = VERIFIERS[family]
    verify(source)  # input must be a self-consistent manifest before touching it

    backup = path.with_name(f"{path.name}.bak-{date.today().isoformat()}")
    if backup.exists():
        raise SystemExit(f"{path}: backup {backup} already exists, refusing")
    backup.write_bytes(raw)

    source["license"] = {
        **license_record,
        "derive_allowed": True,
        "decision": "internal-use",
        "decided_by": DECIDED_BY,
        "decided_at": date.today().isoformat(),
    }
    verify(source)  # snapshot pins untouched; license passes through, must hold
    encoded = json.dumps(source, ensure_ascii=False, indent=2).encode("utf-8")
    path.write_bytes(encoded)

    new_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"[ok]   {path}: derive_allowed -> true")
    print(f"       backup: {backup}")
    print(f"       new source_sha256: {new_sha}")
    return new_sha


def main(argv: list[str]) -> int:
    if not argv:
        raise SystemExit(__doc__)
    for argument in argv:
        repin(Path(argument))
    print(
        "\nRecord the new source_sha256 in the ledger/dry-run record; when a later build references these files,"
        " build.py will freeze them into the new artifact."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
