"""Independent adjacent-binary64 and exact unary-conic audit of552. No production import/optimizer."""
from pathlib import Path
from fractions import Fraction as F
import json,struct,hashlib,time,resource,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(A,B):
 A=A.tocsr();B=B.tocsr();assert A.shape==B.shape and np.array_equal(A.indptr,B.indptr) and np.array_equal(A.indices,B.indices) and np.array_equal(A.data,B.data)
def check(t):
 if time.monotonic()-t>240:raise RuntimeError('240s independent symbolic audit deadline; no continuation')
def neighbor(v,above):
 b=int.from_bytes(struct.pack('>d',v),'big')
 if v==0:b=1 if above else (1<<63)|1
 elif (v>0)==above:b+=1
 else:b-=1
 f=struct.unpack('>d',b.to_bytes(8,'big'))[0];assert np.isfinite(f),'boundary-neighbor range unsupported';return F(f)
def main():
 start=time.monotonic();resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3));assert read(D/'float_hull_preparation_exit_552.json')['actual_exit']==0
 for p,h in read(D/'float_hull_launch_552.json')['source_input_sha256'].items():assert sha(p)==h,p
 P=D/'float_hull_preparation_552';prep=read(P/'preparation.json');assert prep['passed']
 for p,h in read(P/'SHA256.json').items():assert sha(P/p)==h,p
 assert read(D/'box_support_audit_547/independent_audit.json')['passed']
 O=D/'float_hull_audit_553';O.mkdir(exist_ok=False);systems={}
 for role in ['primary','witness']:
  old=D/'box_support_preparation_546'/role;o=P/role;A=sparse.load_npz(old/'matrix.npz').tocsr();a=dict(np.load(old/'arrays.npz'));B=sparse.load_npz(o/'matrix.npz').tocsr();b=dict(np.load(o/'arrays.npz'));bounds=b['bounds'];ep=read(old/'endpoints.json');enc=np.load(o/'encoding.npy');keep=np.load(o/'retained546_positions.npy');rem=np.load(o/'removed546_positions.npy')
  assert np.all(np.isfinite(bounds)) and np.all(bounds[:,0]<=bounds[:,1]);v=np.zeros_like(bounds,dtype=np.uint8);changed=0;worst=F(0)
  with (o/'rounding_ledger.jsonl').open() as log:
   for j in range(len(bounds)):
    check(start);ls,us=map(F,ep[j]);l,u=map(lambda x:F(float(x)),bounds[j]);prev=neighbor(float(bounds[j,0]),False);nxt=neighbor(float(bounds[j,1]),True)
    assert prev<ls<=l and u<=us<nxt and l<=u
    r=json.loads(next(log));assert r[0]==j and F(r[1])==l-ls and F(r[2])==us-u
    for side in [0,1]:
     sign,k,m=map(int,enc[j,side]);assert sign in (-1,0,1) and m>=0 and F(sign*m)*F(2)**k==F(float(bounds[j,side]));v[j,side]=1
    changed+=int(l>ls)+int(u<us);worst=max(worst,l-ls,us-u)
   assert log.read()==''
  assert np.array_equal(rem,np.flatnonzero(np.diff(A.indptr)==1)) and np.array_equal(keep,np.flatnonzero(np.diff(A.indptr)!=1));flags=np.zeros(len(rem),dtype=np.uint8)
  for z,i in enumerate(rem):
   if z%128==0:check(start)
   p=A.indptr[i];j=int(A.indices[p]);coeff=F(float(A.data[p]));side=1 if coeff>0 else 0;unit=F(1 if coeff>0 else -1);rhs=unit*F(float(bounds[j,side]));weight=coeff/unit
   assert weight>0 and weight*unit==coeff and weight*rhs<=F(float(a['rhs'][i]));flags[z]=1
  for j in range(len(bounds)):
   if j%128==0:check(start)
   if np.isfinite(a['bounds'][j,0]):assert F(float(bounds[j,0]))>=F(float(a['bounds'][j,0]))
   if np.isfinite(a['bounds'][j,1]):assert F(float(bounds[j,1]))<=F(float(a['bounds'][j,1]))
  same(B,A[keep]);assert np.array_equal(b['rhs'],a['rhs'][keep]) and np.array_equal(b['cost'],a['cost']) and np.array_equal(b['equality_rhs'],a['equality_rhs']) and np.array_equal(np.load(o/'original_bounds.npy'),a['bounds'])
  assert sha(o/'equality.npz')==sha(old/'equality.npz') and np.array_equal(np.load(o/'bound_owners516.npy'),np.load(old/'bound_owners.npy'))
  assert np.array_equal(np.load(o/'source516_positions.npy'),np.load(old/'retained_source_positions.npy')[keep]) and np.array_equal(np.load(o/'original490_row_ids.npy'),np.load(old/'original_row_ids.npy')[keep])
  s=prep['systems'][role];assert len(rem)==s['singleton_rows_absorbed'] and changed==s['strict_inward_endpoint_changes'] and worst==F(s['maximum_exact_inward_shift']) and list(B.shape)==s['new_shape'] and B.nnz==s['new_nonzeros']
  assert np.all(v) and np.all(flags);np.save(O/(role+'_endpoints_verified.npy'),v);np.save(O/(role+'_singletons_verified.npy'),flags)
  systems[role]=dict(endpoint_adjacency_certificates=int(v.size),singleton_positive_conic_coefficients=len(flags),strict_endpoint_changes=changed,maximum_exact_shift=str(worst),new_shape=list(B.shape),new_nonzeros=int(B.nnz),CSR_RHS_cost_equality_copy_error=0,finite_binary64_feasible_set_equal=True,real_feasible_set_inner_subset=True,original_real_LP_equivalence_claimed=False,old_endpoint_and_mixed_proofs_repeated=False)
 check(start);r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,systems=systems,elapsed_s=time.monotonic()-start,lp_calls=0,small_examples=0,model_calls=0,root_calls=0,finite_maps=0,physical_admission=False,preparation_only=True,no_producer_import=True)
 (O/'independent_audit.json').write_text(json.dumps(r,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
