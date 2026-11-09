"""Saved-only scope index; no matrix, proof, optimizer or model replay."""
from pathlib import Path
import json,datetime
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');read=lambda p:json.loads(Path(p).read_text())
def main():
 assert read(D/'definition_preparation_exit_558.json')['actual_exit']==0 and read(D/'definition_audit_exit_559.json')['actual_exit']==0
 p=read(D/'definition_preparation_558/preparation.json');a=read(D/'definition_audit_559/independent_audit.json');assert p['passed'] and a['passed']
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,preparation_only=True,systems=p['systems'],audit_systems=a['systems'],new_definition_variables=sum(s['definition_variables'] for s in p['systems'].values()),new_definition_equalities=sum(s['definition_variables'] for s in p['systems'].values()),new_signed_source_row_identities=sum(s['signed_source_row_identities'] for s in a['systems'].values()),new_dual_native_coefficient_identities=sum(s['dual_native_coefficient_identities'] for s in a['systems'].values()),new_coordinate_dual_cancellations=sum(s['new_coordinate_dual_cancellations'] for s in a['systems'].values()),net_nonzeros_removed=sum(s['net_nonzeros_removed'] for s in p['systems'].values()),lp_calls=0,small_examples=0,metric_calculations=0,linear_system_solves=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,new_numerical_algorithm=False,new_candidate=False,physical_admission=False,source_original_variables_retained_per_system=78791,exact_real_projection_equivalence=True,finite_binary64_extension_guaranteed=False)
 f=D/'definition_scope_559.json';assert not f.exists();f.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
