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
O = D/'finite_model_step_20261020_v1'
PRIOR = D/'interior_diagnostic_20261020_v1'
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
    assert a['origin']==origin and os.environ['FINITE_MODEL_TOKEN']==a['token']
    assert os.environ['FINITE_MODEL_STAGE']==stage
    check(origin)
    return origin
def load_context():
    import numpy as np
    from scipy import sparse
    from fractions import Fraction as F
    z=dict(np.load(BASE/'reference.npz')); state=dict(np.load(STATE/'state.npz'))
    for key in ['features','actualY','J']:
        assert np.array_equal(z[key],state[key])
    mesh=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'))
    row=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz'))
    eq=dict(np.load(D/'clock_row_preparation_446/equality_arrays.npz'))
    im=read(STATE/'intervals.json'); dom=read(D/'exact_gap_chart_preparation_468/layout.json')
    for key in ['lower','upper','original_lower','original_upper']:
        dom[key]=[F(*v) for v in dom[key]]
    for key in ['total','hover','terminal']: dom[key]=F(*dom[key])
    gl=read(D/'gap_coordinate_preparation_v3_466/layout.json')
    d=dict(z=z,mesh=mesh,row=row,rows=row,eq=eq,dom=dom,intervals=im,
        M=sparse.load_npz(D/'clock_row_preparation_v2_446/all_rows_matrix.npz'),
        E=sparse.load_npz(D/'clock_row_preparation_446/equality_matrix.npz'),
        normal=np.asarray(gl['normal_states']), initial=z['features'],
        active=z['active'],sigma=z['sigma'],control_feature_rows=z['control_feature_rows'],
        varying=z['varying'],lo=z['lower'],hi=z['upper'])
    assert d['M'].shape==(331888,105750)
    assert z['features'].shape==(1410,75) and z['actualY'].shape==(1410,51)
    assert z['J'].shape==(1410,51,24) and len(z['active'])==1365
    assert len(z['control_feature_rows'])==11906 and len(z['varying'])==80155
    assert dom['pivot']==203 and len(dom['independent'])==205
    expected=[['original_control',i] for i in range(11906)]
    expected += [['feature_trust',int(i)] for i in z['varying']]
    expected += [['drop',i] for i in range(206)]
    assert im['owners']==expected and im['trust']==1
    return d
