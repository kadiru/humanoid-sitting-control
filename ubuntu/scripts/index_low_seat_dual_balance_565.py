"""Saved-only scope index; no matrix, proof, optimizer or model replay."""
from pathlib import Path
import json,datetime
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');read=lambda p:json.loads(Path(p).read_text())
def main():
 assert read(D/'dual_balance_preparation_exit_564.json')['actual_exit']==0 and read(D/'dual_balance_audit_exit_565.json')['actual_exit']==0
 p=read(D/'dual_balance_preparation_564/preparation.json');a=read(D/'dual_balance_audit_565/independent_audit.json');assert p['passed'] and a['passed']
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,preparation_only=True,systems=p['systems'],audit_systems=a['systems'],new_transpose_coefficient_copies=sum(s['new_transpose_coefficient_copies'] for s in p['systems'].values()),new_stationarity_coefficient_identities=sum(s['original_stationarity_coefficient_identities'] for s in a['systems'].values()),new_slack_coefficient_identities=sum(s['original_slack_coefficient_identities'] for s in a['systems'].values()),new_scalar_bound_column_identities=sum(s['scalar_bound_column_identities'] for s in a['systems'].values()),new_signed_gap_coordinate_identities=sum(s['signed_gap_coordinate_coefficient_identities'] for s in a['systems'].values()),new_signed_gap_multiplier_identities=sum(s['signed_gap_multiplier_coefficient_identities'] for s in a['systems'].values()),lp_calls=0,small_examples=0,metric_calculations=0,linear_system_solves=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,new_numerical_algorithm=False,new_candidate=False,physical_admission=False,original_coordinate_equality_marginal_slots_per_system=78791,original_scalar_slot=78790,original_domain_unchanged=True,original_primal_is_new_backend_equality_marginal=True,original_primal_is_new_optimizer_primal=False,performance_claim=False)
 f=D/'dual_balance_scope_565.json';assert not f.exists();f.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
