"""Single owner; hard phase deadlines; actual exits and Linux peak RSS."""
import time
import sys
import os
import signal
import subprocess
import fcntl
import resource
from diagnostic_common import D, HERE, O, LIMIT, read, write, workers


def run(stage, script, cap, numerical=False):
    origin = time.monotonic()
    env = dict(os.environ)
    env["PYTHONPATH"] = str(HERE.parent) + (os.pathsep+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    log = (O/(stage+".log")).open("x")
    proc = subprocess.Popen([sys.executable, str(HERE/script), repr(origin)],
                            stdout=log, stderr=subprocess.STDOUT,
                            env=env, start_new_session=True)
    killed = None
    termination = None
    while True:
        waited, status, usage = os.wait4(proc.pid, os.WNOHANG)
        if waited:
            break
        now = time.monotonic()
        reason = None
        deadline = origin + cap
        if numerical:
            returned = read(O/"backend_returned.json") if (O/"backend_returned.json").exists() else None
            if not returned or returned["elapsed_s"] >= 300:
                deadline = min(deadline, origin+300)
                if now >= origin+300:
                    reason = "backend_or_setup_300s_absolute_deadline"
        if now >= origin+cap:
            reason = stage+"_"+str(cap)+"s_absolute_deadline"
        if reason is not None and killed is None:
            killed = reason
            termination = time.monotonic()
            os.killpg(proc.pid, signal.SIGKILL)
        time.sleep(min(.02, max(.001, deadline-time.monotonic())))
    ended = time.monotonic()
    actual_exit = os.waitstatus_to_exitcode(status)
    proc.returncode = actual_exit
    log.close()
    complete = actual_exit == 0 and killed is None and ended-origin < cap
    result = dict(stage=stage, origin=origin, end=ended, elapsed_s=ended-origin,
        configured_seconds=cap, actual_exit=actual_exit, hard_termination_reason=killed,
        termination_signal_sent_at=termination, completed_within_deadline=complete,
        deadline_overshoot_s=max(0., ended-origin-(300 if killed and "300s" in killed else cap)),
        peak_rss_bytes=int(usage.ru_maxrss)*1024, peak_rss_scope="Linux wait4 worker peak RSS",
        address_space_limit_bytes=LIMIT,
        backend_factor_counts=None, backend_linear_solve_counts=None)
    write(O/(stage+"_exit.json"), result)
    print(stage, actual_exit, round(ended-origin, 3), killed or "finished", flush=True)
    return result


def main():
    resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))
    lock = open("/tmp/low_seat_limits_diagnostic_20261020_v1.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not workers({os.getpid(), os.getppid()})
    O.mkdir(exist_ok=False)
    start = time.monotonic()
    pre = run("preflight", "preflight.py", 300)
    if not pre["completed_within_deadline"]:
        write(O/"diagnostic_result.json", dict(stage="prelaunch_failed", candidate_accepted=False,
              primary_calls=0, preflight=pre, no_retry=True))
        return
    numerical = run("numerical", "main_worker.py", 600, numerical=True)
    audit = run("audit", "audit_worker.py", 600)
    ar = read(O/"audit_result.json") if (O/"audit_result.json").exists() else None
    accepted = bool(ar and ar["new_candidate_accepted"] and audit["completed_within_deadline"])
    write(O/"diagnostic_result.json", dict(stage="closed", candidate_accepted=accepted,
        preflight=pre, numerical=numerical, audit=audit,
        stage_elapsed_sum_s=sum(x["elapsed_s"] for x in (pre, numerical, audit)),
        orchestration_elapsed_s=time.monotonic()-start,
        primary_call_marker=(O/"primary/lp/scipy_started.json").exists(),
        interior_calls=0, physical_trials=0, policy_updates=0, holdout_trials=0,
        repaired_controller=False, no_retry=True))
    print("Diagnostic closed; accepted primary:", accepted, flush=True)


if __name__ == "__main__":
    main()
