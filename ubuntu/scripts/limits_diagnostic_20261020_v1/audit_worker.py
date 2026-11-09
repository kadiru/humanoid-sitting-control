"""One independent audit of new output; no producer or optimizer import."""
import time
import sys
START = float(sys.argv[1])
import resource
from diagnostic_common import D, HERE, O, LIMIT, read, write, sha
resource.setrlimit(resource.RLIMIT_AS, (LIMIT, LIMIT))


def main():
    manifest = read(O/"freeze.json")
    # Full live identity, including unchanged producer/auditor source, binds this audit.
    for p, h in manifest["source_input_sha256"].items():
        assert sha(p) == h, p
        assert time.monotonic()-START < 600
    import numpy as np
    from audit_solver import audit_solver
    T = O/"primary/lp"
    execution = read(O/"numerical_exit.json")
    r = read(T/"result.json") if (T/"result.json").exists() else None
    event_lines = [__import__("json").loads(x) for x in (O/"events.jsonl").read_text().splitlines()]
    events = {v["phase"]: v for v in event_lines}
    assert all(v["origin"] == execution["origin"] for v in events.values())
    assert len([v for v in event_lines if v["phase"] == "backend_started"]) <= 1
    actual = (T/"scipy_started.json").exists()
    if actual:
        cfg = read(T/"scipy_started.json")
        opt = cfg["options"]
        assert cfg["method"] == "highs-ipm" and cfg["cold_start"] and not cfg["old_internal_state_used"]
        assert opt["presolve"] is True and opt["maxiter"] == 4096
        assert all(opt[k] == 1e-9 for k in ("primal_feasibility_tolerance", "dual_feasibility_tolerance", "ipm_optimality_tolerance"))
        assert 0 < opt["time_limit"] <= 300
        assert events["input_checks_finished"]["elapsed_s"] + opt["time_limit"] < 300.1
    independent = None
    if r is not None:
        independent = audit_solver(T)
    else:
        # A killed process is terminal. Check any saved new point prefix once.
        if (T/"backend_primal.npy").exists():
            from audit_return import audit_return
            raw = np.load(T/"backend_primal.npy")
            w = np.load(T/"returned_native.npy") if (T/"returned_native.npy").exists() else None
            if raw[-1] == 0:
                independent = dict(returned_graph_point=audit_return(T, raw, w))
        assert not execution["completed_within_deadline"], "missing result without recorded resource/worker failure"
    candidate = bool(r and r["candidate"] and r["status"] == 0 and r["success"])
    accepted = bool(candidate and execution["actual_exit"] == 0 and
                    execution["completed_within_deadline"] and
                    events["backend_returned"]["elapsed_s"] < 300 and
                    r["elapsed_s"] < 600 and independent["passed"])
    if candidate:
        assert independent["returned_graph_point"]["exact_reduced"]["complete"]
        assert independent["returned_graph_point"]["exact_reduced"]["passed"]
        assert len(independent["original_KKT"]) == 6
        assert all(v["passed"] for v in independent["original_KKT"].values())
    result = dict(audit_completed=True, audit_passed=True,
        new_candidate_accepted=accepted, producer_candidate=candidate,
        actual_lp_call=actual, numerical_execution=execution,
        independent_checks=independent, events=events,
        elapsed_s=time.monotonic()-START, physical_admission=False,
        physical_trials=0, policy_updates=0, holdout_trials=0,
        scope="New result/input/timing audit only. No old proofs or optimization replay.")
    assert time.monotonic()-START < 600
    write(O/"audit_result.json", result)
    assert time.monotonic()-START < 600
    print("Independent audit finished; accepted primary:", accepted, flush=True)


if __name__ == "__main__":
    main()
