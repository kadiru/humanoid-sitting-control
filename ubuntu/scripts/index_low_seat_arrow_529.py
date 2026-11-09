"""Saved-result scope only. No matrix assembly/factorization/operator or proof replay."""
from pathlib import Path
import json,datetime
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def main():
 assert not (D/'arrow_operator_scope_529.json').exists()
 prep=read(D/'arrow_operator_preparation_528/preparation.json');audit=read(D/'arrow_operator_audit_529/independent_audit.json');assert prep['passed'] and audit['passed']
 extra=np.load(D/'lifted_feasibility_preparation_490/reference.npz')['extra_rows'];scope={}
 for role in ['primary','witness']:
  p=np.load(D/'arrow_operator_preparation_528'/role/'partition.npz');ret=np.load(D/'torque_projection_preparation_522'/role/'projection_ledger.npz')['retained_columns'];original=ret[p['promoted']];types,counts=np.unique(extra[original-10540]%75,return_counts=True)
  scope[role]=dict(**prep['systems'][role],promoted_feature_indices={str(int(k)):int(n) for k,n in zip(types,counts)},audit_errors=audit['systems'][role]['errors'])
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,systems=scope,total_new_weighted_gram_cholesky_blocks=sum(x['local_cholesky_factorizations'] for x in scope.values()),total_new_local_two_rhs_solves=sum(x['local_triangular_rhs_solves'] for x in scope.values()),new_algorithm_unused_operator_examples=2,LP_columns_unchanged=56951,separator_operator_dimension=33746,old_mechanics_inverse_or_proof_repeated=False,LP_or_Schur_equation_solved=False,new_physical_path=False,latest_actual_model_construction=484,latest_bounded_execution=524,last_sealed_closure=527,physical_admission=False,task_complete=False,automation_status='ACTIVE')
 (D/'arrow_operator_scope_529.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
