"""Construct the new lossless local/separator representation and two operator examples once."""
from low_seat_arrow_operator_528 import *
import datetime
O=D/'arrow_operator_preparation_528'
def main():
 launch=read(D/'arrow_operator_launch_528.json');assert read(D/'arrow_operator_freeze_exit_528.json')['actual_exit']==0
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 O.mkdir(exist_ok=False);extra=np.load(D/'lifted_feasibility_preparation_490/reference.npz')['extra_rows'];results={}
 for role in ['primary','witness']:
  src=D/'torque_projection_preparation_522'/role;out=O/role;out.mkdir();A=sparse.load_npz(src/'matrix.npz').tocsr();ret=np.load(src/'projection_ledger.npz')['retained_columns']
  B,C,p=partition(A,ret,extra);sparse.save_npz(out/'local.npz',B);sparse.save_npz(out/'separator.npz',C);np.savez_compressed(out/'partition.npz',**p)
  r=operator_example(A,B,C,p,out)
  r.update(shape=list(A.shape),source_nnz=A.nnz,local_nnz=B.nnz,separator_nnz=C.nnz,initial_local=int(np.sum(p['initial_labels']>=0)),promoted_columns=len(p['promoted']),cross_state_rows=len(p['cross_rows']),local_columns=len(p['local']),separator_columns=len(p['separator']),retained_all_rows=True,changed_lp_coefficients=False)
  results[role]=r;print(json.dumps(dict(role=role,**r)),flush=True)
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,specification=SPEC,systems=results,lp_calls=0,model_calls=0,root_calls=0,finite_maps=0,new_actual_path=False,physical_admission=False)
 (O/'preparation.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2))
 print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
