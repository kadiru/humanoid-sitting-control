"""Paths and metadata only; no scientific computation."""
from pathlib import Path
import hashlib
import json
import os
import time

D = Path("/data/humanoid-sitting-perception/low_seat_contact_01")
R = Path("/home/kadir/humanoid-sitting")
HERE = Path(__file__).resolve().parent
O = D / "limits_diagnostic_20261020_v1"
LIMIT = 40 * 1024**3


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def event(phase, origin, **extra):
    value = dict(phase=phase, monotonic=time.monotonic(), origin=origin, **extra)
    value["elapsed_s"] = value["monotonic"] - origin
    with (O / "events.jsonl").open("a") as f:
        f.write(json.dumps(value) + "\n")
    if phase == "backend_returned":
        write(O / "backend_returned.json", value)


def workers(excluded):
    found = []
    for p in Path("/proc").glob("[0-9]*/cmdline"):
        if int(p.parent.name) in excluded:
            continue
        try:
            args = p.read_bytes().decode().split("\0")
        except (OSError, UnicodeError):
            continue
        if args and any(x in Path(args[0]).name.lower() for x in ("python", "kit", "isaac")):
            if any(x in " ".join(args) for x in ("humanoid-sitting", "low_seat", "IsaacLab")):
                found.append(dict(pid=int(p.parent.name), args=args))
    return found
