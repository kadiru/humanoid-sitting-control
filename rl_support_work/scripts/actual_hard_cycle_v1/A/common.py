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
