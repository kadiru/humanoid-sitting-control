"""Saved-only index of538/539 exact capped-maximum preparation; no arithmetic replay."""
from pathlib import Path
import json,datetime
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def main():
 p=read(D/'capped_max_preparation_538/preparation.json');a=read(D/'capped_max_audit_539/independent_audit.json')
 assert p['passed'] and a['passed'] and read(D/'capped_max_audit_exit_539.json')['actual_exit']==0
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,systems=p['systems'],audit_systems=a['systems'],source_rows_verified=sum(x['rows_verified'] for x in a['systems'].values()),native_coefficients_verified=sum(x['native_coefficients_verified'] for x in a['systems'].values()),scalar_signs_verified=sum(x['scalar_signs_verified'] for x in a['systems'].values()),new_small_proximal_examples=len(a['new_small_proximal_kkt_checks']),new_small_fenchel_dual_lift_examples=len(a['new_small_fenchel_dual_checks']),new_functional_formulation=True,new_complete_numerical_algorithm=False,native_directions=78790,native_auxiliary_torques_retained=21840,scalar_epigraph_eliminated_per_system=1,inequality_rows_omitted=0,full_size_proximal_updates=0,lp_calls=0,linear_solves=0,gram_factorizations=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,new_candidate=False,new_actual_path=False,physical_admission=False,task_complete=False,limits='Exact-real capped-maximum/conjugate reformulation of516 ORIGINAL representatives; all native directions and signed constraints retained. No522 torque projection or504 transform. Prepared witness ceiling remains illustrative, not an actual result. No optimizer, numerical whole-path wrapper, speedup or feasibility certificate. All original490/finite/actual-state gates remain.')
 q=D/'capped_max_scope_539.json';assert not q.exists();q.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['systems','audit_systems']},indent=2),flush=True)
if __name__=='__main__':main()
