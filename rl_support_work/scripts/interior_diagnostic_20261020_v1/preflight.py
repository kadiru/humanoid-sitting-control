"""Hash/identity/resource freeze only; no scientific imports or calculations."""
import os
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from common import (BASE, CAPS, D, HERE, LIMIT, O, PRIOR, PRIMARY_SHA, R, ROWS,
                    STATE, check_time, inventory, phase, read, sha, workers, write)


def main(origin):
    resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))
    spec = phase(origin, "preflight")
    assert not workers({os.getpid(), os.getppid()})
    required = {
        PRIOR / "summary.json", PRIOR / "SHA256.json",
        PRIOR / "primary/lp/witness.npy", PRIOR / "freeze.json",
        D / "lifted_feasibility_audit_491/independent_audit.json",
        D / "torque_path_audit_525/independent_audit.json", ROWS,
        BASE / "preparation.json", BASE / "reference.npz",
        BASE / "intervals.json", BASE / "all_rows_chain.npz",
        BASE / "consistency_chain.npz", BASE / "interval_matrix.npz",
        BASE / "feature_basis.npz", BASE / "normal_drop_chain.npz",
        D / "balance_path_launch_598.json", D / "balance_path_598/SHA256.json",
        R / "docs/LOW_SEAT_CONTACT_PROTOCOL.md",
        R / "docs/LOW_SEAT_CONTACT_STATUS.md",
    }
    required.update(BASE / "witness" / name for name in
                    ("arrays.npz", "matrix.npz", "equality.npz", "scope.json"))
    required.update(STATE / name for name in
                    ("state.npz", "A.npz", "C.npz", "H.npz", "intervals.json"))
    assert required <= {Path(p) for p in spec["input_sha256"]}
    assert spec["input_sha256"][str(PRIOR / "primary/lp/witness.npy")] == PRIMARY_SHA
    inventory(spec, origin, 300)
    # Preserve the entire accepted primary and inherited evidence, without executing it.
    preserved = []
    for root in (PRIOR, D / "balance_path_598"):
        manifest = read(root / "SHA256.json")
        for relative, digest in manifest.items():
            assert sha(root / relative) == digest, relative
            check_time(origin, 300)
        preserved.append(dict(path=str(root), manifest_sha256=sha(root / "SHA256.json"),
                              files=len(manifest)))
    for name in ("prior_hashes.json", "pressure_quotient_evidence_preservation_603.json",
                 "pressure_quotient_supplemental_603.json",
                 "supplemental_post603_readonly_authoring_0643.json"):
        path = D / name
        assert str(path) in spec["input_sha256"]
        records = read(path)
        for original, digest in records.items():
            assert sha(original) == digest, original
            check_time(origin, 300)
        preserved.append(dict(path=str(path), files=len(records),
                              manifest_sha256=sha(path)))
    previous_freeze = read(PRIOR / "freeze.json")["source_input_sha256"]
    for path, digest in previous_freeze.items():
        assert sha(path) == digest, path
        check_time(origin, 300)
    # Full inherited source/input manifest remains binding. No new vector from it is used.
    old = read(D / "balance_path_launch_598.json")["source_input_sha256"]
    for path, digest in old.items():
        assert sha(path) == digest, path
        check_time(origin, 300)
    assert sha(PRIOR / "summary.json") == "5c05524a5f40bd90fe497c40fc53aa529a1b761e07b0fb93a4b00026db1ea72d"
    assert sha(PRIOR / "SHA256.json") == "6aaee5f547c57eeb0f37974a0d7527d4809c65c2ec9d21dc23df60195a795391"
    assert read(D / "lifted_feasibility_audit_491/independent_audit.json")["passed"]
    assert read(D / "torque_path_audit_525/independent_audit.json")["passed"]
    remaining = 300-(time.monotonic()-origin)
    assert remaining > 0
    result = subprocess.run([sys.executable, "-m", "py_compile",
                             *map(str, sorted(HERE.glob("*.py")))],
                            capture_output=True, text=True, timeout=remaining)
    write(O / "compile.json", dict(actual_exit=result.returncode,
          stdout=result.stdout, stderr=result.stderr))
    assert result.returncode == 0
    free = shutil.disk_usage(D).free
    assert free >= 10 * 1024**3
    assert not workers({os.getpid(), os.getppid()})
    check_time(origin, 300)
    write(O / "freeze.json", dict(passed=True, contract_sha256=sha(HERE / "launch_contract.json"),
          source_input_sha256=spec["input_sha256"], runtime_sha256=spec["runtime_sha256"],
          source_sha256=spec["source_sha256"], preserved=preserved,
          independent_source_review=spec["source_review"],
          original_598_files=len(old), previous_primary_freeze_files=len(previous_freeze),
          remote_compile_actual_exit=0, disk_free_bytes=free, caps_seconds=CAPS,
          scientific_calculations=0, elapsed_s=time.monotonic()-origin))


if __name__ == "__main__":
    main(float(sys.argv[1]))
