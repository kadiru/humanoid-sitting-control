"""LOCAL process-control fixtures only. No scientific imports, arrays or arithmetic."""
from pathlib import Path
import datetime
import hashlib
import importlib
import json
import os
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path("/Users/kadir/humanoid-sitting-control")
SOURCE = ROOT / "rl_support_work/scripts/interior_diagnostic_20261020_v1"
OUT = ROOT / "validated_low_seat_contact_01/interior_supervisor_review_20261020_1030"


def main():
    OUT.mkdir(exist_ok=False)
    sys.path.insert(0, str(SOURCE))
    supervisor = importlib.import_module("supervise")
    real_time = time
    results = []

    def exercise(name, script, stage="numerical", delayed_poll=False, interrupt=False):
        folder = OUT / name
        folder.mkdir()
        (folder / "fixture.py").write_text(script)
        supervisor.O = supervisor.HERE = folder
        supervisor.CAPS = dict(preflight=.5, numerical=.5, audit=.5)
        supervisor.BACKEND_SECONDS = .2
        if delayed_poll:
            supervisor.time = SimpleNamespace(monotonic=real_time.monotonic,
                                               sleep=lambda _: real_time.sleep(.4))
        elif interrupt:
            def interrupted_sleep(_):
                raise KeyboardInterrupt("local fixture interruption")
            supervisor.time = SimpleNamespace(monotonic=real_time.monotonic,
                                               sleep=interrupted_sleep)
        else:
            supervisor.time = real_time
        raised = False
        try:
            record = supervisor.run(stage, "fixture.py")
        except KeyboardInterrupt:
            assert interrupt
            raised = True
            record = json.loads((folder / (stage+"_exit.json")).read_text())
        try:
            os.kill(record["worker_pid"], 0)
        except ProcessLookupError:
            no_live_child = True
        else:
            no_live_child = False
        assert no_live_child
        results.append(dict(name=name, record=record, no_live_child=True,
                            interruption_propagated=raised))
        return record

    good = exercise("completed_worker", "pass\n", stage="preflight")
    assert good["actual_exit"] == 0 and good["completed_within_deadline"]
    bad = exercise("nonzero_worker", "raise SystemExit(7)\n", stage="preflight")
    assert bad["actual_exit"] == 7 and not bad["completed_within_deadline"]
    hung = exercise("backend_deadline", "import time\ntime.sleep(5)\n")
    assert hung["actual_exit"] == -9 and not hung["completed_within_deadline"]
    assert hung["hard_termination_reason"] == "setup_or_backend_deadline"
    assert hung["effective_completion_deadline_seconds"] == .2
    marker = (
        "import json,sys,time,os\nfrom pathlib import Path\n"
        "origin=float(sys.argv[1])\n"
        "p=Path(__file__).parent/'backend_returned.json'\n"
        "temp=p.with_suffix('.tmp')\n"
        "temp.write_text(json.dumps(dict(origin=origin,elapsed_s=time.monotonic()-origin)))\n"
        "os.replace(temp,p)\n")
    total = exercise("validation_total_deadline", marker+"time.sleep(5)\n")
    assert total["actual_exit"] == -9 and not total["completed_within_deadline"]
    assert total["hard_termination_reason"] == "numerical_total_deadline"
    assert total["effective_completion_deadline_seconds"] == .5
    late = exercise("late_exit_between_polls",
                    "import time\ntime.sleep(.25)\n", delayed_poll=True)
    assert late["actual_exit"] == 0 and not late["completed_within_deadline"]
    assert late["effective_completion_deadline_seconds"] == .2
    assert late["deadline_overshoot_s"] > 0
    cancelled = exercise("interrupted_supervisor", "import time\ntime.sleep(5)\n",
                         interrupt=True)
    assert cancelled["actual_exit"] == -9 and not cancelled["completed_within_deadline"]
    assert cancelled["hard_termination_reason"] == "supervisor_interrupted_or_failed"
    supervisor.time = real_time
    compile_result = subprocess.run(
        [sys.executable, "-m", "py_compile", *map(str, sorted(SOURCE.glob("*.py")))],
        capture_output=True, text=True)
    assert compile_result.returncode == 0, compile_result.stderr
    result = dict(passed=True, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope="LOCAL process-control fixtures only; deliberately short fixture clocks; production caps unchanged",
        compile_actual_exit=compile_result.returncode, fixtures=results,
        scientific_runs=0, remote_uploads=0, production_launch_authorized=False,
        production_caps_seconds=dict(preflight=300, numerical=600, backend_absolute=300, audit=600),
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted(SOURCE.glob("*.py"))})
    (OUT / "review.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(dict(passed=True, fixture_count=len(results), all_children_stopped=True,
                         compile_actual_exit=0, scientific_runs=0, output=str(OUT))))


if __name__ == "__main__":
    main()
