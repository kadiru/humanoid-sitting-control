"""Metadata only. This draft cannot run without a separate approved launch contract."""
from pathlib import Path
import hashlib
import json
import os
import time

HERE = Path(__file__).resolve().parent
D = Path("/data/humanoid-sitting-perception/low_seat_contact_01")
R = Path("/home/kadir/humanoid-sitting")
O = D / "interior_diagnostic_20261020_v1"
PRIOR = D / "limits_diagnostic_20261020_v1"
BASE = D / "lifted_feasibility_preparation_490"
STATE = D / "torque_path_524/major_00"
ROWS = D / "clock_row_preparation_v2_446/row_arrays.npz"
LIMIT = 40 * 1024**3
CAPS = {"preflight": 300, "numerical": 600, "audit": 600}
PRIMARY_SHA = "7715ec610c7769a1e2ba5ba2d2b0955f0d073317231fe74136c9447608f00d4d"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_time(origin, seconds):
    if time.monotonic() - origin >= seconds:
        raise TimeoutError("Declared absolute stage deadline reached")


def contract():
    authorization = read(HERE / "authorization.json")
    if authorization.get("execution_approved") is not True:
        raise PermissionError("Separate interior diagnostic approval is pending")
    spec = read(HERE / "launch_contract.json")
    assert spec["independent_source_review_complete"] is True
    assert spec["authorization_sha256"] == sha(HERE / "authorization.json")
    assert spec["proposal_sha256"] == sha(HERE / "proposal.json")
    assert spec["caps_seconds"] == CAPS
    assert spec["backend_absolute_seconds"] == 300
    assert spec["address_space_bytes"] == LIMIT
    assert spec["counts"] == dict(cold_interior_solves=1, primary_repeats=0,
        finite_maps=0, model_calls=0, physical_trials=0, PPO_updates=0,
        holdout_trials=0)
    assert spec["input_sha256"] and spec["runtime_sha256"]
    assert set(spec["source_sha256"]) == {
        p.name for p in HERE.iterdir()
        if p.is_file() and p.name != "launch_contract.json"
    }
    for name, digest in spec["source_sha256"].items():
        assert sha(HERE / name) == digest, name
    return spec


def phase(origin, stage):
    # A direct invocation cannot start scientific work outside supervision.
    assert os.environ.get("INTERIOR_DIAGNOSTIC_STAGE") == stage
    admission = read(O / (stage + "_admission.json"))
    assert admission["origin"] == origin
    assert admission["token"] == os.environ.get("INTERIOR_DIAGNOSTIC_TOKEN")
    spec = contract()
    if stage != "preflight":
        freeze = read(O / "freeze.json")
        assert freeze["passed"] is True
        assert freeze["contract_sha256"] == sha(HERE / "launch_contract.json")
    check_time(origin, CAPS[stage])
    return spec


def workers(excluded):
    found = []
    for item in Path("/proc").glob("[0-9]*/cmdline"):
        if int(item.parent.name) in excluded:
            continue
        try:
            args = item.read_bytes().decode().split("\0")
        except (OSError, UnicodeError):
            continue
        if args and any(s in Path(args[0]).name.lower()
                        for s in ("python", "kit", "isaac")):
            if any(s in " ".join(args)
                   for s in ("humanoid-sitting", "low_seat", "IsaacLab")):
                found.append({"pid": int(item.parent.name), "args": args})
    return found


def inventory(spec, origin, seconds):
    for path, digest in {**spec["input_sha256"],
                         **spec["runtime_sha256"]}.items():
        assert sha(path) == digest, path
        check_time(origin, seconds)


def event(name, origin, **details):
    value = dict(event=name, origin=origin, elapsed_s=time.monotonic()-origin,
                 **details)
    with (O / "events.jsonl").open("a") as output:
        output.write(json.dumps(value, allow_nan=False) + "\n")
    if name == "backend_returned":
        write(O / "backend_returned.json", value)
    return value
