"""Small shared IO utilities; no simulator import or launch side effects."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/kadir/humanoid-sitting')
DATA = Path('/data/humanoid-sitting-perception/low_seat_contact_01')
DT = .02
RATE = .15
STEP = RATE * DT
FILTER = .1
MARGIN = .03
OBS = 447  # Original336 plus111 latched initial sensor values; no truth/case ID.


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()
