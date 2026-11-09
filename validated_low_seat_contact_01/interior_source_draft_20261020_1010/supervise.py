"""One supervised interior diagnostic, blocked pending a separate approved freeze."""
import fcntl
import os
import resource
import secrets
import signal
import subprocess
import sys
import time
from common import CAPS, HERE, LIMIT, O, contract, read, workers, write


def run(stage, script):
    origin = time.monotonic()
    token = secrets.token_hex(24)
    write(O / (stage+"_admission.json"), dict(origin=origin, token=token))
    env = dict(os.environ, INTERIOR_DIAGNOSTIC_STAGE=stage,
               INTERIOR_DIAGNOSTIC_TOKEN=token)
    # Do not inherit an optimized Python mode that would remove scientific assertions.
    env.pop("PYTHONOPTIMIZE", None)
    log = (O / (stage+".log")).open("x")
    process = subprocess.Popen([sys.executable, str(HERE / script), repr(origin)],
                               stdout=log, stderr=subprocess.STDOUT,
                               env=env, start_new_session=True)
    killed, sent_at = None, None
    cap = CAPS[stage]
    while True:
        waited, status, usage = os.wait4(process.pid, os.WNOHANG)
        if waited:
            break
        elapsed = time.monotonic()-origin
        reason = None
        if stage == "numerical":
            marker = O / "backend_returned.json"
            backend_done = marker.exists() and read(marker)["elapsed_s"] < 300
            if not backend_done and elapsed >= 300:
                reason = "setup_or_backend_300s_deadline"
        if elapsed >= cap:
            reason = stage+"_total_deadline"
        if reason is not None and killed is None:
            killed, sent_at = reason, time.monotonic()
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        time.sleep(.01)
    end = time.monotonic()
    code = os.waitstatus_to_exitcode(status)
    process.returncode = code
    log.close()
    completed = code == 0 and killed is None and end-origin < cap
    if stage == "numerical" and (O / "backend_returned.json").exists():
        completed = completed and read(O / "backend_returned.json")["elapsed_s"] < 300
    effective_cap = 300 if killed == "setup_or_backend_300s_deadline" else cap
    record = dict(stage=stage, origin=origin, end=end, elapsed_s=end-origin,
                  configured_seconds=cap, actual_exit=code,
                  completed_within_deadline=completed, hard_termination_reason=killed,
                  termination_signal_sent_at=sent_at,
                  deadline_overshoot_s=max(0., end-origin-effective_cap),
                  address_space_limit_bytes=LIMIT,
                  peak_rss_bytes=int(usage.ru_maxrss)*1024)
    write(O / (stage+"_exit.json"), record)
    print(stage, code, round(end-origin, 3), killed or "finished", flush=True)
    return record


def main():
    # Must fail here before remote paths, process ownership or numerical imports are touched.
    contract()
    assert sys.platform == "linux" and sys.flags.optimize == 0
    resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))
    lock = open("/tmp/low_seat_interior_diagnostic_20261020_v1.lock", "a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not workers({os.getpid(), os.getppid()})
    O.mkdir(exist_ok=False)
    stages = [run("preflight", "preflight.py")]
    if stages[0]["completed_within_deadline"]:
        stages.append(run("numerical", "producer.py"))
        stages.append(run("audit", "independent.py"))
    ar = read(O / "audit_result.json") if (O / "audit_result.json").exists() else None
    accepted = bool(len(stages) == 3 and all(s["completed_within_deadline"] for s in stages)
                    and ar and ar["passed"] and ar["candidate_accepted"])
    write(O / "diagnostic_result.json", dict(candidate_accepted=accepted, stages=stages,
          elapsed_stage_sum_s=sum(s["elapsed_s"] for s in stages),
          cold_solve_started=(O / "scipy_started.json").exists(),
          no_retry=True, primary_repeats=0, finite_maps=0, model_calls=0,
          physical_trials=0, PPO_updates=0, holdout_trials=0, repaired_controller=False))
    print("Interior diagnostic closed; finite-margin direction accepted:", accepted, flush=True)


if __name__ == "__main__":
    main()
