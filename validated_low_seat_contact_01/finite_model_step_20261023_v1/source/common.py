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
O = D/'finite_model_step_20261023_v1'
PRIOR = D/'interior_certificate_revalidation_20261023_v1'
CURRENT = D/'changed_state_primary_20261020_v1'
ACCEPTED = D/'finite_model_step_20261020_v1'
PRIMARY = D/'full_minimax_budget_review_20261023_v1'
INTERIOR = D/'interior_after_minimax_20261023_v1'
BASE = D/'lifted_feasibility_preparation_490'
STATE = D/'torque_path_524/major_00'
LIMIT = 40*1024**3
CAP = 45.0
PRODUCER_END = 30.0

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
    authorization=read(HERE/'authorization.json')
    assert authorization['execution_approved'] is True
    c=read(HERE/'contract.json')
    assert c['execution_approved'] is True and c['authorization_sha256']==sha(HERE/'authorization.json')
    assert c['solver_calls']==0 and c['producer_hard_deadline_s']==PRODUCER_END
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
def identical(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def bind_current(reference,state,accepted_features,accepted_y,accepted_J):
    import numpy as np
    z=dict(reference)
    for key,accepted in [('features',accepted_features),('actualY',accepted_y),('J',accepted_J)]:
        identical(state[key],accepted)
        assert np.isfinite(state[key]).all()
        z[key]=state[key]
    return z
def load_context():
    import numpy as np
    from scipy import sparse
    from fractions import Fraction as F
    reference=dict(np.load(BASE/'reference.npz',allow_pickle=False))
    state=dict(np.load(CURRENT/'state.npz',allow_pickle=False))
    accepted_f=np.load(ACCEPTED/'mapped.npz',allow_pickle=False)['features']
    accepted_a=dict(np.load(ACCEPTED/'actual.npz',allow_pickle=False))
    accepted_j=dict(np.load(ACCEPTED/'accepted_J.npz',allow_pickle=False))
    identical(accepted_a['changed'],accepted_j['changed'])
    z=bind_current(reference,state,accepted_f,accepted_a['Y'],accepted_j['J'])
    mesh=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'))
    row=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz'))
    eq=dict(np.load(D/'clock_row_preparation_446/equality_arrays.npz'))
    im=read(CURRENT/'intervals.json'); dom=read(D/'exact_gap_chart_preparation_468/layout.json')
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
    identical(z['features'][mesh['fixed']],reference['features'][mesh['fixed']])
    assert im['drops']==read(ACCEPTED/'map.json')['drop_state']
    assert read(CURRENT/'input_proof.json')['new_state_full_input_proved']
    assert read(ACCEPTED/'audit.json')['passed'] and read(ACCEPTED/'result.json')['trial_merit_accepted']
    assert read(PRIMARY/'ceiling.json')['mu']==read(ACCEPTED/'decision.json')['summary']['phi']
    expected=[['original_control',i] for i in range(11906)]
    expected += [['feature_trust',int(i)] for i in z['varying']]
    expected += [['drop',i] for i in range(206)]
    assert im['owners']==expected and im['trust']==1
    return d
