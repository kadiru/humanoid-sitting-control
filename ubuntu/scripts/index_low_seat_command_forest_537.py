"""Index saved536/537 negative evidence only; no matrix or arithmetic replay."""
from pathlib import Path
import json,datetime
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def main():
 p=read(D/'command_forest_preparation_536/preparation.json');a=read(D/'command_forest_audit_537/independent_audit.json')
 assert p['passed'] and a['passed'] and a['removed_row_certificates_empty']
 for r in ['primary','witness']:
  assert p['systems'][r]['implied_rows']==0
  assert a['systems'][r]['negative_decision_audit']['exact_tested_decisions']==a['systems'][r]['negative_decision_audit']['negative_endpoint_decisions']==9920
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,systems=p['systems'],audit_systems=a['systems'],new_graph_systems=2,new_exact_path_decisions=19840,new_negative_path_decisions=19840,new_path_cap_decisions=68672,new_proved_implications=0,new_removed_rows=0,new_lp_variables_removed=0,independent_negative_coefficient_additions=sum(s['negative_decision_audit']['coefficient_additions'] for s in a['systems'].values()),successful_reduction=False,full_projected_matrices_unchanged=True,empty_success_proof_scope='No removed rows; all_proof_rows_explicitly_retained and zero positive-certificate additions describe EMPTY successful-proof sets. Separately audited19840 negative path decisions are actual new evidence.',negative_scope='The fixed forest and fixed unary choices failed every tested endpoint implication.68672 candidate paths exceeded128-edge arithmetic cap and were retained untested. No global nonredundancy/infeasibility claim. No cap extension, alternate forest or unary-selection retry.',lp_calls=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,physical_admission=False,new_candidate=False,new_solver=False,task_complete=False)
 q=D/'command_forest_scope_537.json';assert not q.exists();q.write_text(json.dumps(out,indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['systems','audit_systems']},indent=2),flush=True)
if __name__=='__main__':main()
