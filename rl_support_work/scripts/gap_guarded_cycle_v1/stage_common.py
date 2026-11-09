"""One stage clock, source-bound inputs and audited predecessor manifests."""
import json
import os
import sys
import time
from pathlib import Path
from policy import authorize, read, sha, ROOT, LIMIT, STAGES

R = Path('/home/kadir/humanoid-sitting')
D = Path('/data/humanoid-sitting-perception/low_seat_contact_01')
CYCLE = D/'gap_guarded_cycle_v1'
CURRENT = D/'targeted_gap_primary_20261023_v1'
ACCEPTED = D/'finite_model_step_20261023_v1'
BASE = D/'lifted_feasibility_preparation_490'
STATE = D/'torque_path_524/major_00'
PRIMARY = CYCLE/'A'
INTERIOR = CYCLE/'B'

def write(p, x):
    p = Path(p)
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')
    os.replace(tmp, p)

def workers(excluded):
    answer = []
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name) in excluded:
            continue
        try:
            args = p.read_bytes().decode().split('\0')
        except (OSError, UnicodeError):
            continue
        if args and any(x in Path(args[0]).name.lower() for x in ('python', 'kit', 'isaac')):
            if any(x in ' '.join(args) for x in ('humanoid-sitting', 'low_seat', 'IsaacLab')):
                answer.append({'pid': int(p.parent.name), 'args': args})
    return answer

def bind_predecessor(folder, origin, check):
    # Hash-and-identity inheritance only, no re-evaluation of old points/proofs.
    outer = read(folder/'execution_exit.json')
    assert outer['passed'] and outer['no_retry']
    assert sha(folder/'manifest.json') == outer['manifest_sha256']
    for name, expected in read(folder/'manifest.json').items():
        assert Path(name).name == name and sha(folder/name) == expected, name
        check(origin)
    result = read(folder/'result.json')
    assert result['execution_and_audit_passed'] and result['candidate_accepted']
    audit = read(folder/'audit.json')
    assert audit['passed'] and audit['candidate_accepted']
    assert result['audit_sha256'] == sha(folder/'audit.json')

def verify_static(spec, origin, check):
    for p, expected in {**spec['input_sha256'], **spec['runtime_sha256']}.items():
        assert sha(p) == expected, p
        check(origin)

def inherited_input(origin, check):
    a = read(CURRENT/'input_proof.json')
    b = read(CURRENT/'target_input_proof.json')
    assert a['passed'] and a['new_state_full_input_proved'] and b['passed']
    assert read(ACCEPTED/'audit.json')['passed']
    assert read(ACCEPTED/'result.json')['trial_merit_accepted']
    check(origin)

def solver_options(remaining):
    assert 0 < remaining <= 420
    return dict(time_limit=remaining, maxiter=4096, presolve=True,
        primal_feasibility_tolerance=1e-9, dual_feasibility_tolerance=1e-9,
        ipm_optimality_tolerance=1e-9)
