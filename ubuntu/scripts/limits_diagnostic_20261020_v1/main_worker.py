"""Exactly one diagnostic primary, starting its clock before imports and preparation."""
import time
import sys
START = float(sys.argv[1])
import resource
import shutil
from diagnostic_common import D, HERE, O, LIMIT, read, write, sha, event
resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))


def main():
    assert read(O/"preflight_exit.json")["actual_exit"] == 0
    manifest = read(O/"freeze.json")
    event("main_preparation_started", START)
    for p, h in manifest["source_input_sha256"].items():
        assert sha(p) == h, p
        assert time.monotonic()-START < 300, "setup exhausted backend deadline"
    target = O/"primary"
    target.mkdir()
    for name in ["matrix.npz", "arrays.npz", "equality.npz", "stage.json"]:
        source = D/"balance_path_598/major_00/primary"/name
        shutil.copy2(source, target/name)
        assert sha(source) == sha(target/name)
    out = target/"lp"
    out.mkdir()
    write(out/"clock_origin.json", dict(parent_monotonic_origin=START,
        parent_preparation_elapsed_s=time.monotonic()-START, all_new_preparation_charged=True))
    event("main_preparation_finished", START)
    sys.argv = [str(HERE/"solve_diagnostic.py"), str(out), repr(START)]
    import solve_diagnostic
    solve_diagnostic.main()
    assert time.monotonic()-START < 600, "numerical result finished beyond deadline"


if __name__ == "__main__":
    main()
