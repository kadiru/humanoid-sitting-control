"""Finite furniture-validation IO. No simulator side effects."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/kadir/humanoid-sitting')
HERE = Path(__file__).resolve().parent
ASSETS = ROOT/'experiments/furniture_test_20261028_v1'
TRAIN = ROOT/'experiments/recovery_imitation_sitting_20261028_v3'
PPO = ROOT/'outputs/recovery_imitation_sitting_20261028_v3'
OUT = ROOT/'outputs/furniture_validation_20261028_v2'
GPU = 'GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
PY = ROOT/'IsaacLab/.venv/bin/python'

def read(p): return json.loads(Path(p).read_text())
def write(p, value):
    p = Path(p)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(p)
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

