"""Saved-only accounting for518; no model, matrix rebuild, map or solve."""
from pathlib import Path
import json,datetime
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');read=lambda p:json.loads(Path(p).read_text())
def main():
    assert read(D/'quotient_path_exit_518.json')['actual_exit']==0
    P=D/'quotient_path_518';r=read(P/'results.json');h=[json.loads(x) for x in (P/'history.jsonl').read_text().splitlines()];stages=[];trials=[]
    for m in h:
        for role in ['primary','interior']:
            if role in m:stages.append(dict(major=m['major'],trust=m['trust'],role=role,**m[role]))
        trials += [dict(major=m['major'],**t) for t in m['trials']]
    final=np.load(P/'final.npz');seed=np.load(D/'free_gap_path_472/initial.npz')
    scope=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,actual_scipy_linprog_calls=sum(x['actual_lp_call'] for x in stages),assembled_subproblems=len(stages),worker_actual_exits=[x['worker_actual_exit'] for x in stages],worker_wall_cap_hits=sum(x['worker_wall_timeout'] for x in stages),transformed_returned_witnesses=sum(x['transformed_candidate'] for x in stages),primary_calls=sum(x['role']=='primary' for x in stages),interior_calls=sum(x['role']=='interior' for x in stages),time_limits_without_witness=sum(x['status']==1 and not x['candidate'] for x in stages),returned_witnesses=sum(x['candidate'] for x in stages),certificate_attempts=len(trials),finite_maps=r['finite_maps'],actual_new_trial_evaluations=r['actual_trial_evaluations'],accepted_solver_iterates=r['accepted_solver_iterates'],actual_X_exact472=bool(np.array_equal(final['X'],seed['X'])),actual_Y_exact472=bool(np.array_equal(final['Y'],seed['Y'])),actual_J_exact472=bool(np.array_equal(final['J'],seed['J'])),auxiliary_Y_exact_actual_Y=bool(np.array_equal(final['features'][:,24:],final['Y'])),inherited_quotient_systems=r["inherited_quotient_systems"],new_quotient_systems=r["new_quotient_systems"],new_quotient_proof_rows=r["new_quotient_proof_rows"],stage_records=stages,trial_records=trials,model_calls=0,optimizer_calls=0,physical_admission=False)
    assert scope['actual_scipy_linprog_calls']==r['actual_lp_calls']
    p=D/'quotient_saved_scope_521.json';assert not p.exists();p.write_text(json.dumps(scope,indent=2));print(json.dumps(scope,indent=2),flush=True)
if __name__=='__main__':main()
