"""Independent exact row-reconstruction/dual-column audit of558; no producer or optimizer import."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time,resource,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(t):
 if time.monotonic()-t>240:raise RuntimeError('240s independent algebra audit deadline; no continuation')
def same(A,B):
 A=A.tocsr();B=B.tocsr();assert A.shape==B.shape and np.array_equal(A.indptr,B.indptr) and np.array_equal(A.indices,B.indices) and np.array_equal(A.data,B.data)
def row(A,i):
 s,e=A.indptr[i:i+2];return {int(j):F(float(v)) for j,v in zip(A.indices[s:e],A.data[s:e])}
def combine(a,b,sign):
 out=a.copy()
 for j,v in b.items():out[j]=out.get(j,F(0))+sign*v
 return {j:v for j,v in out.items() if v}
def main():
 start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3));assert read(D/'definition_preparation_exit_558.json')['actual_exit']==0
 for p,h in read(D/'definition_launch_558.json')['source_input_sha256'].items():assert sha(p)==h,p
 P=D/'definition_preparation_558';prep=read(P/'preparation.json');assert prep['passed']
 for p,h in read(P/'SHA256.json').items():assert sha(P/p)==h,p
 assert read(D/'box_support_audit_547/independent_audit.json')['passed'] and read(D/'exact_row_quotient_audit_517/independent_audit.json')['passed']
 O=D/'definition_audit_559';O.mkdir(exist_ok=False);systems={}
 for role in ['primary','witness']:
  src=D/'box_support_preparation_546'/role;o=P/role;A=sparse.load_npz(src/'matrix.npz').tocsr();E=sparse.load_npz(src/'equality.npz').tocsr();a=dict(np.load(src/'arrays.npz'));B=sparse.load_npz(o/'matrix.npz').tocsr();EE=sparse.load_npz(o/'equality.npz').tocsr();b=dict(np.load(o/'arrays.npz'));ids=np.load(src/'original_row_ids.npy');pairs=np.load(o/'pairs.npy');decisions=np.load(o/'decisions.npy');n=A.shape[1];k=len(pairs);ne=E.shape[0]
  assert n==78791 and k>0 and B.shape==(A.shape[0],n+k) and EE.shape==(ne+k,n+k)
  assert A.has_canonical_format and B.has_canonical_format and EE.has_canonical_format
  assert np.isfinite(B.data).all() and np.isfinite(EE.data).all() and np.all(B.data!=0) and np.all(EE.data!=0)
  index={int(v):j for j,v in enumerate(ids)};assert len(index)==len(ids);decisionflags=np.zeros(68250,dtype=np.uint8);zindex=0;terms=0;rowflags=np.zeros((k,2),dtype=np.uint8);dualflags=np.zeros(k,dtype=np.uint8);affected=np.zeros(A.shape[0],dtype=bool)
  for i in range(68250):
   if i%64==0:check(start)
   p=index.get(331495+i);q=index.get(399745+i)
   if p is None or q is None:assert decisions[i]==0;decisionflags[i]=1;continue
   plus=row(A,p);minus=row(A,q);native={j:v for j,v in plus.items() if j<78790};other={j:v for j,v in minus.items() if j<78790}
   assert native=={j:-v for j,v in other.items()}
   if len(native)<4:assert decisions[i]==1;decisionflags[i]=1;continue
   assert decisions[i]==2 and np.array_equal(pairs[zindex],np.array([i,p,q,n+zindex,len(native)]));z=n+zindex
   definition=row(EE,ne+zindex);assert definition==(native|{z:F(-1)})
   rp=row(B,p);rm=row(B,q);assert rp==({z:F(1)}|({78790:plus[78790]} if 78790 in plus else {}));assert rm==({z:F(-1)}|({78790:minus[78790]} if 78790 in minus else {}))
   assert combine(rp,definition,F(1))==plus and combine(rm,definition,F(-1))==minus
   # Coefficients of independent y_plus/y_minus in the dual substitution nu=y_plus-y_minus.
   for j,v in native.items():assert F(1)*definition[j]==plus[j] and F(-1)*definition[j]==minus[j]
   assert rp[z]+definition[z]==0 and rm[z]-definition[z]==0
   rowflags[zindex]=1;dualflags[zindex]=1;decisionflags[i]=1;affected[p]=True;affected[q]=True;terms+=len(native);zindex+=1
  assert zindex==k and np.all(decisionflags) and np.all(rowflags) and np.all(dualflags)
  same(B[~affected,:n],A[~affected]);assert B[~affected,n:].nnz==0;same(EE[:ne,:n],E);assert EE[:ne,n:].nnz==0
  assert np.array_equal(b['rhs'],a['rhs']) and np.array_equal(b['cost'][:n],a['cost']) and np.all(b['cost'][n:]==0)
  assert np.array_equal(b['bounds'][:n],a['bounds']) and np.isneginf(b['bounds'][n:,0]).all() and np.isposinf(b['bounds'][n:,1]).all()
  assert np.array_equal(b['equality_rhs'][:ne],a['equality_rhs']) and np.all(b['equality_rhs'][ne:]==0)
  assert np.array_equal(np.load(o/'original_row_ids.npy'),ids) and np.array_equal(np.load(o/'source516_positions.npy'),np.load(src/'retained_source_positions.npy'))
  s=prep['systems'][role];assert k==s['definition_variables'] and terms==s['definition_native_coefficients'] and A.nnz+E.nnz-B.nnz-EE.nnz==terms-3*k==s['net_nonzeros_removed']
  assert list(B.shape)==s['extended_shape'] and B.nnz==s['extended_inequality_nonzeros'] and EE.nnz==s['extended_equality_nonzeros']
  np.save(O/(role+'_decisions_verified.npy'),decisionflags);np.save(O/(role+'_source_rows_verified.npy'),rowflags);np.save(O/(role+'_dual_columns_verified.npy'),dualflags)
  systems[role]=dict(pair_ID_decisions_verified=len(decisionflags),lifted_pairs=k,signed_source_row_identities=2*k,dual_native_coefficient_identities=2*terms,new_coordinate_dual_cancellations=2*k,native_coefficients_stored_once=terms,net_nonzeros_removed=terms-3*k,unaltered_inequality_rows=int((~affected).sum()),new_shape=list(B.shape),new_equality_shape=list(EE.shape),original_domain_cost_RHS_E_ID_copy_error=0,exact_real_projection_and_objective_equivalence=True,finite_binary64_extension_claimed=False,old_517_547_proofs_repeated=False)
 check(start);r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,systems=systems,elapsed_s=time.monotonic()-start,lp_calls=0,small_examples=0,model_calls=0,root_calls=0,finite_maps=0,physical_admission=False,preparation_only=True,no_producer_import=True,no_candidate_dual_or_primal_evaluated=True)
 (O/'independent_audit.json').write_text(json.dumps(r,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
