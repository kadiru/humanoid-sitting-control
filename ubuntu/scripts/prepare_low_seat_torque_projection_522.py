"""Produce the two NEW saved-system exact existential projections once."""
from low_seat_torque_projection_522 import *
import datetime
O=D/'torque_projection_preparation_522'
def main():
 launch=read(D/'torque_projection_launch_522.json');assert launch['compile_actual_exit']==0
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 O.mkdir(exist_ok=False);ref=np.load(D/'lifted_feasibility_preparation_490/reference.npz');ex=ref['extra_rows'];el=10540+np.flatnonzero((ex%75>=55)&(ex%75<71));assert len(el)==21840
 results={}
 for role in ['primary','witness']:
  src=D/'exact_row_quotient_preparation_516'/role/'original';out=O/role;out.mkdir();A=sparse.load_npz(src/'matrix.npz');arr=dict(np.load(src/'arrays.npz'));E=sparse.load_npz(D/'lifted_feasibility_preparation_490'/role/'equality.npz');assert E[:,el].nnz==0
  results[role]=project(A,arr,el,out);ret=np.load(out/'projection_ledger.npz')['retained_columns'];sparse.save_npz(out/'equality.npz',E[:,ret]);np.save(out/'source_row_ids.npy',np.load(src.parent/'row_ids.npy'))
 examples=arithmetic_examples();(O/'new_projection_examples.json').write_text(json.dumps(examples,indent=2))
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,specification=SPEC,systems=results,new_small_arithmetic_examples=2,inherited517_proof_not_repeated=True,new_actual_path=False,lp_calls=0,model_calls=0,finite_maps=0,physical_admission=False)
 (O/'preparation.json').write_text(json.dumps(r,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
