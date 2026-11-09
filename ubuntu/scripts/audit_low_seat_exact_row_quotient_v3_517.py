"""Independent exact implication proof for every omitted saved inequality."""
from pathlib import Path
from fractions import Fraction
import datetime,json,hashlib
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'exact_row_quotient_preparation_516';O=D/'exact_row_quotient_audit_517'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def row(M,i):
 s,e=M.indptr[i:i+2];v=M.data[s:e];c=M.indices[s:e];mask=v!=0
 pairs=sorted(zip(c[mask].tolist(),v[mask].tolist()))
 assert all(pairs[k][0]<pairs[k+1][0] for k in range(len(pairs)-1))
 return np.asarray([x for x,_ in pairs]),np.asarray([x for _,x in pairs])
def check(M,b,reps,ids):
 retained=set(map(int,ids));checked=np.zeros(M.shape[0],dtype=np.uint8);strict=np.zeros(M.shape[0],dtype=np.uint8)
 taut=0;cross=0;nonunit=0
 for i,j in enumerate(reps):
  ci,vi=row(M,i)
  if j<0:
   assert len(vi)==0 and Fraction(float(b[i]))>=0;taut+=1;checked[i]=1;continue
  assert int(j) in retained
  if j==i:checked[i]=1;continue
  cj,vj=row(M,int(j));assert len(vi) and np.array_equal(ci,cj)
  ap=Fraction(float(vi[0]));bp=Fraction(float(vj[0]));ratio=ap/bp;assert ratio>0
  # Independent rational cross-products, without the production gcd keys.
  for x,y in zip(vi,vj):
   assert Fraction(float(x))*bp==Fraction(float(y))*ap,(i,int(j));cross+=1
  slack=Fraction(float(b[i]))-ratio*Fraction(float(b[j]));assert slack>=0,(i,int(j))
  strict[i]=int(slack>0);nonunit+=int(ratio!=1);checked[i]=1
 assert np.all(checked)
 return checked,strict,dict(checked_rows=len(reps),tautologies=taut,nontrivial_implications=int(np.count_nonzero((reps>=0)&(reps!=np.arange(len(reps))))),strict_rhs_implications=int(strict.sum()),nonunit_positive_scales=nonunit,exact_coefficient_cross_products=cross)
def main():
 assert read(D/'exact_row_quotient_preparation_exit_516.json')['actual_exit']==1
 assert read(D/'exact_row_quotient_finalization_exit_516.json')['actual_exit']==0
 for p,h in read(D/'exact_row_quotient_staged_launch_516.json')['preserved_source_output_sha256'].items():assert sha(p)==h,p
 for p,h in read(D/'exact_row_quotient_launch_v2_516.json')['source_input_sha256'].items():assert sha(p)==h,p
 for p,h in read(P/'SHA256.json').items():assert sha(P/p)==h,p
 O.mkdir(exist_ok=False);details={}
 for mode in ['primary','witness']:
  ids=np.load(P/mode/'row_ids.npy');assert np.array_equal(ids,np.unique(ids));own=[];details[mode]={}
  for name,folder in [('original','lifted_feasibility_preparation_490'),('transformed','mixed_coordinate_preparation_504')]:
   src=D/folder/mode;M=sparse.load_npz(src/'matrix.npz').tocsr();a=dict(np.load(src/'arrays.npz'));q=dict(np.load(P/mode/(name+'_ledger.npz')))
   checked,strict,d=check(M,a['rhs'],q['representative'],ids)
   actual_pivots=[];actual_lengths=[]
   for i in range(M.shape[0]):
    _,v=row(M,i);actual_pivots.append(v[0] if len(v) else 0.);actual_lengths.append(len(v))
   assert np.array_equal(q['pivot'],actual_pivots) and np.array_equal(q['support_length'],actual_lengths)
   assert np.array_equal(q['representatives'],np.unique(q['representative'][q['representative']>=0]));own.append(q['representatives'])
   assert np.array_equal(q['tautology_rows'],np.flatnonzero(q['representative']<0))
   assert np.array_equal(q['contradiction_rows'],np.flatnonzero((q['support_length']==0)&(a['rhs']<0)))
   saved=sparse.load_npz(P/mode/name/'matrix.npz').tocsr();wanted=M[ids]
   assert np.array_equal(saved.indptr,wanted.indptr) and np.array_equal(saved.indices,wanted.indices) and np.array_equal(saved.data,wanted.data)
   z=dict(np.load(P/mode/name/'arrays.npz'));assert np.array_equal(z['rhs'],a['rhs'][ids])
   for k in ['cost','bounds','equality_rhs']:assert np.array_equal(z[k],a[k])
   eq=sparse.load_npz(src/'equality.npz');assert eq.shape==(1366,78791)
   np.savez_compressed(O/(mode+'_'+name+'_proof.npz'),checked=checked,strict_rhs=strict)
   d.update(selected_rows=len(ids),omitted_rows=M.shape[0]-len(ids),selected_nnz=saved.nnz,coefficient_rhs_copy_error=0,equality_source_sha256=sha(src/'equality.npz'))
   details[mode][name]=d;print(json.dumps(dict(mode=mode,representation=name,**d)),flush=True)
  assert np.array_equal(ids,np.union1d(*own))
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,systems=details,synthetic_fixture_passed=False,synthetic_fixture_saved_output_available=False,synthetic_fixture_repeated=False,scope_excludes_failed_unsaved_synthetic_fixture=True,all_omitted_rows_exactly_implied_in_both_systems=True,full_original_and_transformed_numerical_witness_still_required=True,production_grouping_rerun=False,lp_calls=0,model_calls=0,root_calls=0,finite_maps=0,new_actual_path=False,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
