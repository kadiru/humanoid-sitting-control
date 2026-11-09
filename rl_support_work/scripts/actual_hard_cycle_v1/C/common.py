"""Stage-specific frozen paths; authorization precedes scientific imports."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from stage_common import *
NAME=HERE.name
O=CYCLE/NAME
CAP,SOLVE_END,PRODUCER_END,AUDIT_END=STAGES[NAME]
PRIOR=CURRENT if NAME in ('A','B') else INTERIOR

def check(origin):
    if time.monotonic()-origin >= CAP: raise TimeoutError('Shared stage deadline')
def event(name,origin,**kw):
    check(origin)
    with (O/'events.jsonl').open('a') as f:
        f.write(json.dumps(dict(event=name,elapsed_s=time.monotonic()-origin,**kw))+'\n')
def contract():
    f=authorize(); c=read(HERE/'contract.json')
    assert f['stage_contract_sha256'][NAME]==sha(HERE/'contract.json')
    assert c['total_seconds']==CAP and c['address_space_bytes']==LIMIT
    assert c['ready'] and c['worker_address_space_bytes']==WORKER_LIMIT
    assert c['helper_address_space_bytes']==HELPER_LIMIT and c['output_bound_bytes']==OUTPUT_LIMITS[NAME]
    assert c['backend_soft_deadline_s']==SOLVE_END and c['producer_hard_deadline_s']==PRODUCER_END
    assert c['audit_hard_deadline_s']==AUDIT_END
    return c
def admission(stage):
    assert sys.flags.optimize==0
    import resource
    assert resource.getrlimit(resource.RLIMIT_AS)[0]==WORKER_LIMIT
    origin=float(sys.argv[1]); a=read(O/'admission.json')
    assert origin==a['origin'] and os.environ['ACTUAL_HARD_CYCLE_TOKEN']==a['token']
    assert os.environ['ACTUAL_HARD_CYCLE_STAGE']==stage
    assert a['contract_sha256']==sha(HERE/'contract.json')
    check(origin); return origin

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
    d=dict(baseline_actual_rows=accepted_a['actual_rows'],z=z,mesh=mesh,row=row,rows=row,eq=eq,dom=dom,intervals=im,
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
