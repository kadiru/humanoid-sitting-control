"""CPU postprocessing of the October 29 recordings. No simulator dependencies."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path("/home/kadir/humanoid-sitting")
RUN = "furniture_perception_repair_20261029_v1"
SAVED = ROOT / "outputs" / RUN
SOURCE = ROOT / "experiments" / RUN
ASSETS = ROOT / "experiments/furniture_test_20261028_v1"
OUT = ROOT / "outputs/furniture_saved_review_20261104_v1"
FLAGS = ["fall", "forbidden contact", "torque", "impact", "joint limit"]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish(path, value):
    """Exclusive, atomic publication. An existing result must match exactly."""
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    with tempfile.NamedTemporaryFile(dir=path.parent) as tmp:
        tmp.write(payload)
        tmp.flush()
        try:
            os.link(tmp.name, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise ValueError(f"Conflicting evidence at {path}")


def checked_auditor(source, freeze):
    path = source / "audit.py"
    for file in (path, source / "common.py"):
        assert sha(file) == freeze[str(file)]
    spec = importlib.util.spec_from_file_location("saved_policy_audit", path)
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(source))
    try:
        sys.modules.pop("common", None)
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module.physical


def audit_trial(case, freeze):
    folder = SAVED / ("robot_" + case["id"])
    manifest = read(folder / "manifest.json")
    assert case == manifest["case"]
    result = read(folder / "results.json")
    row, = result["episodes"]
    assert result["success_criteria_unchanged"] and manifest["action_limits_unchanged"]
    assert row["id"] == case["id"] and row["steps"] == 1000 and row["duration_s"] == 20
    assert row["full_standing_start"] and result["warmup_transitions"] == 25
    mapping = read(folder / "contact_mapping.json")
    assert mapping["thresholds"] == dict(surface_distance_m=.015, perimeter_inset_m=.01)
    assert len(mapping["frames"]) == 1025
    assert mapping["validated"] == all(f["mapping_valid"] for f in mapping["frames"])
    with np.load(folder / "trajectory.npz", allow_pickle=False) as z:
        d = dict(z)
    assert all(v.shape[:2] == (1000, 1) and np.isfinite(v).all() for v in d.values())
    source = Path(manifest["source"])
    pred = checked_auditor(source, freeze)(d)
    warm, = read(folder / "startup.json")["warm_failure"]
    for key in ("failure", "legal", "no_slip"):
        np.testing.assert_array_equal(pred[key], d[key], err_msg=key)
    for key in ("alive", "good"):
        np.testing.assert_array_equal(pred[key] & (not warm), d[key], err_msg=key)
    assert row["clean"] == bool(pred["clean"][0] and not warm)
    assert row["stable"] == bool(pred["stable"][0] and not warm)
    assert row["slip_ticks"] == int((~pred["no_slip"]).sum())
    assert row["illegal_ticks"] == int((~pred["legal"]).sum())
    hold = 0
    for value, held in zip(d["good"][:, 0], d["held_ticks"][:, 0]):
        hold = hold + 1 if value else 0
        assert hold == held
    assert row["final_hold_s"] == .02 * hold
    force = d["physical_force_w"][:, 0]
    np.testing.assert_array_equal(force, d["force"][:, 0])
    weight = d["physical_bodyweight_N"][:, 0]
    active = np.linalg.norm(force, axis=2) > .01 * weight[:, None]
    inside = d["physical_inside_patch"][:, 0]
    frames = mapping["frames"][25:]
    for tick, frame in enumerate(frames):
        valid = True
        for j in range(5):
            labels = frame["mapped"][j]
            normals = np.array([x["normal"] for x in labels])
            single = bool(labels and np.all(np.linalg.norm(normals - normals[0], axis=1) < 1e-5))
            if active[tick, j] and not single:
                valid = False
            interior = single and all(x["valid"] for x in labels)
            if j < 3 or not interior:
                assert inside[tick, j] == interior
        assert valid == frame["mapping_valid"]
    flags = np.stack([
        (d["root_height"] < .28) | (d["physical_trunk_tilt_rad"] > np.deg2rad(55)),
        (d["wrong_force"] > 10) | (active & ~inside).any(-1)[:, None],
        (abs(d["physical_torque"]) > d["physical_torque_scale"] + .001).any(-1),
        d["peak_bw"] > 2, ~d["physical_joint_margin_ok"],
    ], axis=-1)
    np.testing.assert_array_equal(np.logical_or.accumulate(flags, axis=0), d["flags"])
    np.testing.assert_array_equal(flags.any(0)[0], row["physical_failures"])
    # Diagnostic only: substantial normal force on a permitted interior furniture
    # face while the robot is still eligible. Excludes both floor channels.
    normal_load = (force * d["physical_normal_w"][:, 0]).sum(axis=-1).clip(min=0)
    intended = inside & active & d["physical_permitted"][:, 0] & d["physical_rest_patch"][:, 0]
    supported = (normal_load * intended).sum(-1) >= .05 * weight
    supported &= d["alive"][:, 0] & np.array([f["mapping_valid"] for f in frames])
    runs = np.diff(np.flatnonzero(np.diff(np.r_[False, supported, False])) .reshape(-1, 2), axis=1)
    first = np.flatnonzero(d["failure"][:, 0])
    bindings = {n: sha(folder / n) for n in ("trajectory.npz", "results.json", "manifest.json",
                                             "contact_mapping.json", "startup.json")}
    return dict(passed=True, id=row["id"], complete_trials=1, duration_s=20, raw_clean=row["clean"],
                contact_mapping_validated=mapping["validated"], clean=row["clean"] if mapping["validated"] else None,
                physical_failures=row["physical_failures"], failure_types=[n for n, v in zip(FLAGS, row["physical_failures"]) if v],
                final_hold_s=row["final_hold_s"], slip_ticks=row["slip_ticks"], illegal_ticks=row["illegal_ticks"],
                first_recorded_failure_s=.02 * (int(first[0])+1) if len(first) else None, warm_failure=warm,
                max_consecutive_intended_load_before_failure_s=.02 * int(runs.max(initial=0)),
                action_limits_changed=False, success_thresholds_changed=False, trajectory_sha256=bindings["trajectory.npz"],
                inputs=bindings, physical_audit_source=str(source / "audit.py"), physical_audit_sha256=freeze[str(source / "audit.py")],
                mode="Offline reconstruction from recorded forces, poses and mesh labels; warmup validity as recorded.")


def main():
    freeze = read(SOURCE / "freeze.json")
    cases = read(ASSETS / "cases.json")
    assert len({c["id"] for c in cases}) == len(cases) == 14
    trials = [audit_trial(c, freeze) for c in cases]
    fixture = read(SAVED / "body_support/independent_audit.json")
    assert fixture["passed"]
    for path, digest in fixture["files"].items():
        assert sha(Path(path)) == digest
    rows = fixture["rows"]
    assert len(rows) == 42 and {(r["case"], r["round"]) for r in rows} == {(c["id"], j) for c in cases for j in range(3)}
    layouts = []
    for case, trial in zip(cases, trials):
        checks = sorted((r for r in rows if r["case"] == case["id"]), key=lambda r: r["round"])
        depth = read(SAVED / ("robot_" + case["id"]) / "depth_audit.json")
        assert depth["complete"] and depth["case"] == case["id"] and len(depth["frames"]) == 3
        for j, frame in enumerate(depth["frames"]):
            assert frame["input_sha256"] == sha(SAVED / ("robot_" + case["id"]) / f"frame_{j}.npz")
        layouts.append(dict(id=case["id"], furniture=case["title"], approach=case["approach"], task=case["task"],
                            robot=trial, fixture=checks, load_passes=sum(r["physical_support"] for r in checks),
                            on_target_load_passes=sum(r["physical_support"] and r["intended_contact_patch"] for r in checks),
                            depth_geometry=depth["geometry_pass"], plane_gate=depth["original_plane_gate"]))
    OUT.mkdir(exist_ok=True)
    # Fill only the absent reports so the original report's input paths work.
    for trial in trials:
        publish(SAVED / ("robot_" + trial["id"]) / "independent_audit.json", trial)
        dest = OUT / ("robot_" + trial["id"])
        dest.mkdir(exist_ok=True)
        publish(dest / "independent_audit.json", trial)
    summary = dict(source_run=RUN, simulator_steps=0, ppo_updates=0, trials=layouts, fixture_load_N=fixture["load_N"],
                   whole_robot_clean=sum(t["clean"] is True for t in trials),
                   raw_clean=sum(t["raw_clean"] for t in trials), unvalidated_scores=sum(t["clean"] is None for t in trials),
                   fixture_load_passes=sum(r["physical_support"] for r in rows),
                   fixture_on_target_load_passes=sum(r["physical_support"] and r["intended_contact_patch"] for r in rows),
                   fixture_audit_sha256=sha(SAVED / "body_support/independent_audit.json"))
    publish(OUT / "audit_summary.json", summary)
    for r in layouts:
        print(r["furniture"], r["approach"], r["robot"]["clean"], r["robot"]["max_consecutive_intended_load_before_failure_s"], r["load_passes"], r["on_target_load_passes"])
    print({k:v for k,v in summary.items() if k != "trials"})


if __name__ == "__main__":
    main()
