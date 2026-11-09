"""Saved-only index of exact-rational auxiliary output elimination; no science replay."""
from pathlib import Path
import json,hashlib,datetime
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');read=lambda p:json.loads(Path(p).read_text())
def main():
 assert read(D/'affine_output_preparation_exit_578.json')['actual_exit']==0 and read(D/'affine_output_audit_exit_579.json')['actual_exit']==0
 p=read(D/'affine_output_preparation_578/preparation.json');a=read(D/'affine_output_audit_579/independent_audit.json');assert p['passed'] and a['passed'] and p['primary']==a['primary']
 scope=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),preparation_only=True,last_sealed_closure='closure_record_577.json',primary=p['primary'],preparation_elapsed_s=p['elapsed_s'],audit_elapsed_s=a['elapsed_s'],independent_coefficient_products=a['independent_coefficient_products'],independent_RHS_products=a['independent_RHS_products'],independent_pivot_multiplier_identities=a['independent_pivot_multiplier_identities'],exact_copy_error=a['exact_copy_error'],numerical_lp_calls=0,linear_system_solves=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,primal_vectors_constructed=0,analytic_dual_vectors_constructed=0,rounded_coefficients_created=0,physical_admission=False,new_solver=False,new_candidate=False,old_proofs_repeated=False,scientific_execution_recovery=False,scope='Exact rational reduced zero-face system only. Auxiliary47775 removed with exact affine graphs;physicalX retained. No roundednumericmatrix/finite-extension guarantee/primalwitness/actualpath. All64protected/all8unresolved,mainPPO/freshholdout unstarted.')
 out=D/'affine_output_scope_579.json';assert not out.exists();out.write_text(json.dumps(scope,indent=2));print(json.dumps(scope,indent=2))
if __name__=='__main__':main()
