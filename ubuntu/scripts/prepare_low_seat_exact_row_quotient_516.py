"""Saved full-row exact dominance preparation. No LP/model/finite-map call."""
from low_seat_exact_row_quotient_516 import *
import datetime
O=D/'exact_row_quotient_preparation_516'
def main():
 launch=read(D/'exact_row_quotient_launch_516.json');assert launch['compile_actual_exit']==0
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 O.mkdir(exist_ok=False);results={}
 for mode in ['primary','witness']:
  out=O/mode;out.mkdir();records={};union=[]
  for name,folder in [('original','lifted_feasibility_preparation_490'),('transformed','mixed_coordinate_preparation_504')]:
   src=D/folder/mode;M=sparse.load_npz(src/'matrix.npz').tocsr();a=dict(np.load(src/'arrays.npz'))
   q=quotient(M,a['rhs']);np.savez_compressed(out/(name+'_ledger.npz'),**q);union.append(q['representatives'])
   records[name]=dict(source=str(src),full_shape=list(M.shape),full_nnz=M.nnz,positive_ray_groups=len(q['representatives'])-len(q['contradiction_rows']),tautologies=len(q['tautology_rows']),contradictions=len(q['contradiction_rows']),own_representatives=len(q['representatives']))
   print(json.dumps(dict(stage=mode,representation=name,**records[name])),flush=True)
  ids=np.union1d(*union);np.save(out/'row_ids.npy',ids)
  for name,r in records.items():
   src=Path(r['source']);M=sparse.load_npz(src/'matrix.npz').tocsr();a=dict(np.load(src/'arrays.npz'));dest=out/name;dest.mkdir()
   sparse.save_npz(dest/'matrix.npz',M[ids]);np.savez_compressed(dest/'arrays.npz',rhs=a['rhs'][ids],cost=a['cost'],bounds=a['bounds'],equality_rhs=a['equality_rhs'])
   # Equality files are inherited byte-for-byte by explicit reference.
   r.update(selected_rows=len(ids),selected_nnz=M[ids].nnz,omitted_rows=M.shape[0]-len(ids),equality_source=str(src/'equality.npz'),equality_sha256=sha(src/'equality.npz'))
  results[mode]=records
 m,t,b=new_arithmetic_case();q=quotient(m,b);v=quotient(t,b)
 assert q['representative'].tolist()==[1,1,2,3,4,-1,6,1,1]
 assert v['representative'].tolist()==[1,1,2,3,4,-1,6,7,1]
 np.savez_compressed(O/'new_implication_case.npz',original=m.toarray(),transformed=t.toarray(),rhs=b,original_representative=q['representative'],transformed_representative=v['representative'],union=np.union1d(q['representatives'],v['representatives']))
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,specification=SPEC,systems=results,new_paired_implication_fixture=1,full_basis_or_row_transforms_rebuilt=False,lp_calls=0,model_calls=0,finite_maps=0,new_actual_path=False,physical_admission=False)
 (O/'preparation.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
