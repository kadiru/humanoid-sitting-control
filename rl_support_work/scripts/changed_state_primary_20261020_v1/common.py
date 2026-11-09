"""Frozen data access and process admission; no solver or scientific import."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
R = Path('/home/kadir/humanoid-sitting')
D = Path('/data/humanoid-sitting-perception/low_seat_contact_01')
O = D/'changed_state_primary_20261020_v1'
PRIOR = D/'finite_model_step_20261020_v1'
BASE = D/'lifted_feasibility_preparation_490'
STATE = D/'torque_path_524/major_00'
LIMIT = 40*1024**3
CAP = 45.0

def read(p): return json.loads(Path(p).read_text())
def write(p, x):
    p = Path(p)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, p)
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4194304), b''): h.update(block)
    return h.hexdigest()
def check(origin):
    if time.monotonic()-origin >= CAP: raise TimeoutError('Shared 45-second total limit')
def event(name, origin, **kw):
    check(origin)
    with (O/'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(event=name, elapsed_s=time.monotonic()-origin, **kw))+'\n')
def workers(excluded):
    result=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name) in excluded: continue
        try: args=p.read_bytes().decode().split('\0')
        except (OSError, UnicodeError): continue
        if args and any(x in Path(args[0]).name.lower() for x in ['python','kit','isaac']):
            if any(x in ' '.join(args) for x in ['humanoid-sitting','low_seat','IsaacLab']):
                result.append(dict(pid=int(p.parent.name), args=args))
    return result
def contract():
    c=read(HERE/'contract.json')
    assert c['ready'] and c['independent_source_review_complete']
    assert c['total_seconds']==CAP and c['address_space_bytes']==LIMIT
    assert c['source_sha256']
    for name,h in c['source_sha256'].items(): assert sha(HERE/name)==h, name
    return c
def admission(stage):
    assert sys.flags.optimize==0
    origin=float(sys.argv[1]); a=read(O/'admission.json')
    assert a['origin']==origin and os.environ['CHANGED_PRIMARY_TOKEN']==a['token']
    assert os.environ['CHANGED_PRIMARY_STAGE']==stage
    check(origin)
    return origin
