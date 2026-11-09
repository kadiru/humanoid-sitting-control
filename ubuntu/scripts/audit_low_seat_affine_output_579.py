"""Independent exact reverse equality-row reconstruction; no producer/optimizer import."""
import time,resource
START=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def limit():
 if time.monotonic()-START>240:raise TimeoutError('240s independent affine-output audit cap')
def frac(v):
 assert len(v)==2 and type(v[0]) is int and type(v[1]) is int and v[1]>0
 q=F(v[0],v[1]);assert q.numerator==v[0] and q.denominator==v[1];return q
def source_row(M,i):
 lo,hi=M.indptr[i:i+2]
 return {int(M.indices[k]):F(float(M.data[k])) for k in range(lo,hi) if M.data[k]!=0}
def same(a,b):
 assert a.shape==b.shape and a.dtype==b.dtype
 if a.dtype.kind=='f':assert np.array_equal(a.view(np.uint64),b.view(np.uint64))
 else:assert np.array_equal(a,b)
def same_csr(a,b):
 assert a.shape==b.shape;same(a.indptr,b.indptr);same(a.indices,b.indices);same(a.data,b.data)
def main():
 assert read(D/'affine_output_preparation_exit_578.json')['actual_exit']==0
 launch=read(D/'affine_output_launch_578.json')
 for p,h in launch['source_input_sha256'].items():assert digest(p)==h,p
 new=D/'affine_output_preparation_578';src=D/'zero_face_preparation_572'
 for p,h in read(new/'SHA256.json').items():assert digest(new/p)==h,p
 assert read(D/'zero_face_audit_573/independent_audit.json')['passed']
 A=sparse.load_npz(src/'matrix.npz').tocsr();E=sparse.load_npz(src/'equality.npz').tocsr();a=dict(np.load(src/'arrays.npz'));maps=dict(np.load(new/'maps.npz'));b=dict(np.load(new/'arrays.npz'))
 eliminated=[];pivot=[]
 for block in range(1365):
  eliminated.extend(range(10540+50*block+15,10540+50*block+50))
  pivot.extend([1366+50*block+i for i in list(range(14,46))+[47,48,49]])
 eset=set(eliminated);pset=set(pivot);retained=[j for j in range(78791) if j not in eset];eqretained=[j for j in range(69617) if j not in pset]
 for name,x in [('retained_columns',retained),('eliminated_columns',eliminated),('pivot_rows',pivot),('retained_equality_rows',eqretained)]:same(np.array(x,dtype=np.int64),maps[name])
 assert len(retained)==31016 and len(eliminated)==47775 and len(eqretained)==21842 and retained[-1]==78790
 assert np.all(a['cost'][eliminated]==0) and np.isneginf(a['bounds'][eliminated,0]).all() and np.isposinf(a['bounds'][eliminated,1]).all()
 graph=[];gcoeff=0;flags=np.zeros(47775,dtype=np.uint8)
 with (new/'affine_graph.jsonl').open() as f:
  for j,line in enumerate(f):
   if j%128==0:limit()
   assert j<len(eliminated);col,row,d,h,terms=json.loads(line);assert col==eliminated[j] and row==pivot[j]
   original=source_row(E,row);aux={k for k in original if k in eset};assert aux=={col};assert frac(d)==original[col] and frac(d)!=0;assert frac(h)==F(float(a['equality_rhs'][row]))
   loaded={};previous=-1
   for k,num,den in terms:
    assert type(k) is int and k>previous and k not in eset;previous=k;loaded[k]=frac([num,den]);assert loaded[k]!=0
   assert loaded=={k:v for k,v in original.items() if k!=col}
   graph.append((original,F(float(a['equality_rhs'][row]))));gcoeff+=len(original);flags[j]=1
 assert len(graph)==47775
 ER=sparse.load_npz(new/'retained_equality.npz').tocsr();assert E[eqretained][:,eliminated].nnz==0;same_csr(ER,E[eqretained][:,retained].tocsr())
 same(a['cost'][retained],b['cost']);same(a['bounds'][retained],b['bounds']);same(a['equality_rhs'][eqretained],b['equality_rhs']);assert set(b)=={'cost','bounds','equality_rhs'}
 same(np.load(src/'original_row_ids.npy'),np.load(new/'original_row_ids.npy'))
 eloc={k:j for j,k in enumerate(eliminated)};rowflags=np.zeros(A.shape[0],dtype=np.uint8);products=0;rhsproducts=0;lambda_checks=0;nnz=0;constants=0;maxbits=0;forwardproducts=0
 with (new/'rational_inequalities.jsonl').open() as f:
  for i,line in enumerate(f):
   if i%64==0:limit()
   assert i<A.shape[0];index,terms,rhs,mults=json.loads(line);assert index==i
   source=source_row(A,i);restored={};previous=-1
   for k,num,den in terms:
    assert type(k) is int and previous<k<len(retained);previous=k;v=frac([num,den]);assert v!=0;restored[retained[k]]=v;maxbits=max(maxbits,v.numerator.bit_length(),v.denominator.bit_length())
   nnz+=len(restored);constants+=not bool(restored);value=frac(rhs);maxbits=max(maxbits,value.numerator.bit_length(),value.denominator.bit_length())
   expected={eloc[k] for k in source if k in eset};seen=set()
   # Reverse reconstruction: reduced row + signed multiples of original equality premises.
   for j,num,den in mults:
    assert type(j) is int and j in expected and j not in seen;seen.add(j);lam=frac([num,den]);assert lam!=0
    original,h=graph[j];col=eliminated[j];assert lam*original[col]==source[col];lambda_checks+=1
    for k,v in original.items():restored[k]=restored.get(k,F(0))+lam*v;products+=1
    value+=lam*h;rhsproducts+=1;forwardproducts+=len(original)-1
   assert seen==expected
   restored={k:v for k,v in restored.items() if v};assert restored==source,(i,'coefficients');assert value==F(float(a['rhs'][i])),(i,'RHS')
   rowflags[i]=1
 assert len(rowflags)==265967 and np.all(rowflags==1);limit()
 q=read(new/'preparation.json')['primary']
 expected=dict(source_columns=78791,reduced_columns=len(retained),eliminated_auxiliary_columns=len(eliminated),retained_physical_X_coordinates=20475,source_inequality_rows=int(A.shape[0]),reduced_inequality_rows=int(A.shape[0]),source_inequality_nonzeros=int(A.nnz),exact_reduced_inequality_nonzeros=nnz,source_equality_rows=int(E.shape[0]),reduced_equality_rows=int(ER.shape[0]),source_equality_nonzeros=int(E.nnz),retained_equality_nonzeros=int(ER.nnz),graph_nonzeros=gcoeff,pivot_incidence_checks=len(eliminated),eliminated_inequality_occurrences=lambda_checks,exact_multiplier_divisions=lambda_checks,exact_coefficient_products=forwardproducts,exact_RHS_products=rhsproducts,constant_rows_retained=constants,largest_fraction_bits=maxbits,reduced_scalar_column=31015,original_scalar_column=78790,rows_discarded=0,original_bounds_changed=False,rounded_coefficients_created=0,primal_graph_evaluations=0,old_proofs_repeated=0)
 assert q==expected,(q,expected)
 out=D/'affine_output_audit_579';out.mkdir(exist_ok=False);np.save(out/'pivot_graph_verified.npy',flags);np.save(out/'inequality_reconstruction_verified.npy',rowflags);np.save(out/'retained_equality_rows_verified.npy',np.asarray(eqretained,dtype=np.int64));np.save(out/'retained_columns_verified.npy',np.asarray(retained,dtype=np.int64))
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,arithmetic_passed=True,preparation_only=True,primary=q,independent_coefficient_products=products,independent_RHS_products=rhsproducts,independent_pivot_multiplier_identities=lambda_checks,new_graph_coefficient_copies=gcoeff,exact_copy_error=0,elapsed_s=time.monotonic()-START,producer_import=False,numerical_optimizer_calls=0,linear_system_solves=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,primal_vectors_constructed_or_evaluated=0,analytic_dual_vectors_constructed=0,physical_admission=False,execution_recovery=False,old_proofs_replayed=False,scope='Exact rational projection equivalence to572zeroFACE only;47775 eliminated auxiliaryoutputs,physicalX retained. No roundednumericmatrix,universalbinary64extension,primalwitness,convergence orphysicalclaim.')
 (out/'independent_audit.json').write_text(json.dumps(r,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):digest(p) for p in out.rglob('*') if p.is_file()},indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
