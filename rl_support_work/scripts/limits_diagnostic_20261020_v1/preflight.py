"""Bounded preservation, compilation and source/input/audit freeze. No scientific imports."""
import time
import sys
import os
import resource
import subprocess
import datetime
import shutil
from diagnostic_common import D, R, HERE, O, LIMIT, read, write, sha, workers

START = float(sys.argv[1])
resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))


def check():
    assert time.monotonic() - START < 300, "prelaunch deadline"


def main():
    assert read(HERE / "authorization.json")["execution_approved"] is True
    assert not (O / "freeze.json").exists()
    active = workers({os.getpid(), os.getppid()})
    assert not active, active
    check()
    old = read(D / "balance_path_launch_598.json")
    assert sha(D / "balance_path_launch_598.json") == "9737d66275ca694e0260114915fd8a8759a929f2f6d6fd09d0880a6ab06e0677"
    # Exact source transformations are an auditable restriction to one primary;
    # the inherited arithmetic functions and all tolerance gates are retained.
    derivations = read(HERE / "source_derivations.json")
    for new, spec in derivations.items():
        source = R / "scripts" / spec["source"]
        assert sha(source) == spec["source_sha256"]
        text = source.read_text()
        for edit in spec["replacements"]:
            assert text.count(edit["before"]) == edit["count"]
            text = text.replace(edit["before"], edit["after"])
        assert text == (HERE / new).read_text()
        assert sha(HERE / new) == spec["derived_sha256"]
    compile_cmd = [sys.executable, "-m", "py_compile"] + [str(p) for p in sorted(HERE.glob("*.py"))]
    compiled = subprocess.run(compile_cmd, capture_output=True, text=True, timeout=max(.01, 300-(time.monotonic()-START)))
    write(O / "remote_compile.json", dict(actual_exit=compiled.returncode, stdout=compiled.stdout, stderr=compiled.stderr))
    assert compiled.returncode == 0
    check()
    hashes = {}
    checks = []
    for name in ["prior_hashes.json", "pressure_quotient_evidence_preservation_603.json",
                 "pressure_quotient_supplemental_603.json", "supplemental_post603_readonly_authoring_0643.json"]:
        m = read(D / name)
        for p, h in m.items():
            assert sha(p) == h, p
            check()
        checks.append(dict(manifest=name, sha256=sha(D/name), files=len(m), passed=True))
    for p, h in old["source_input_sha256"].items():
        assert sha(p) == h, p
        hashes[p] = h
        check()
    for rel, h in read(D/"balance_path_598/SHA256.json").items():
        p = D/"balance_path_598"/rel
        assert sha(p) == h, str(p)
        check()
    # Freeze all inherited original results as evidence; no old result is a start vector.
    for name in ["balance_path_launch_598.json", "balance_path_598/SHA256.json",
                 "pressure_quotient_evidence_preservation_603.json",
                 "pressure_quotient_supplemental_603.json", "handoff_prepared_603.txt",
                 "closure_record_601.json", "status.json"]:
        hashes[str(D/name)] = sha(D/name)
    for name in ["matrix.npz", "arrays.npz", "equality.npz", "stage.json"]:
        p = D/"balance_path_598/major_00/primary"/name
        assert sha(p) == sha(D/"torque_path_524/major_00/primary"/name), name
        hashes[str(p)] = sha(p)
    for p in sorted(HERE.iterdir()):
        if p.is_file():
            hashes[str(p)] = sha(p)
    for name, expected in [
        ("LOW_SEAT_CONTACT_PROTOCOL.md", "0df7fa154b64c99f7c992c6a6317e87e62adac7ab31410daa140f742d01a4415"),
        ("LOW_SEAT_CONTACT_STATUS.md", "7c2c8fd221ca7d021f7ffdf582665fe68fd0523d9b34c8671cb3812bfd24b7c6")]:
        p = R/"docs"/name
        assert sha(p) == expected
        hashes[str(p)] = expected
    assert shutil.disk_usage(D).free > 10 * 1024**3, "insufficient output storage reserve"
    assert not workers({os.getpid(), os.getppid()})
    check()
    write(O/"freeze.json", dict(
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_input_sha256=hashes, preserved=checks, remote_compile_actual_exit=0,
        original_manifest_sha256=sha(D/"balance_path_launch_598.json"),
        original_specification=old["specification"], proposal=read(HERE/"proposal.json"),
        authorization=read(HERE/"authorization.json"), source_review=read(HERE/"review.json"),
        source_derivations_verified=True, original_arithmetic_unchanged=True,
        input_copy_identity_verified=True, workers=[],
        disk_free_bytes=shutil.disk_usage(D).free, elapsed_s=time.monotonic()-START,
        permitted_primary_calls=1, permitted_interior_calls=0,
        physical_trials=0, policy_updates=0, holdout_trials=0))
    check()
    print("Prelaunch identity, preservation, compilation and freeze complete.", flush=True)


if __name__ == "__main__":
    main()
