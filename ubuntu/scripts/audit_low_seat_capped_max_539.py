"""Independent source-row, scalar-dual and exact proximal-KKT audit. No production import/solve."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert read(D/'capped_max_preparation_exit_538.json')['actual_exit']==0
 for p,h in read(D/'capped_max_launch_538.json')['source_input_sha256'].items():assert sha(p)==h,p
 prep=D/'capped_max_preparation_538'
 for p,h in read(prep/'SHA256.json').items():assert sha(prep/p)==h,p
 old=read(D/'exact_row_quotient_audit_517/independent_audit.json');assert old['passed'] and old['all_omitted_rows_exactly_implied_in_both_systems']
 out=D/'capped_max_audit_539';out.mkdir(exist_ok=False);systems={}
 for role in ['primary','witness']:
  src=D/'exact_row_quotient_preparation_516'/role/'original';A=sparse.load_npz(src/'matrix.npz').tocsr();arr=dict(np.load(src/'arrays.npz'));z=dict(np.load(prep/role/'arrays.npz'));n=A.shape[1]-1;e=F(float(arr['cost'][-1]));assert e in (1,-1)
  H=sparse.load_npz(prep/role/'hard.npz').tocsr();S=sparse.load_npz(prep/role/'coupled.npz').tocsr();E=sparse.load_npz(D/'lifted_feasibility_preparation_490'/role/'equality.npz');En=sparse.load_npz(prep/role/'native_equality.npz')
  assert E.shape==(1366,n+1) and En.shape==(1366,n) and E.nnz==En.nnz==0
  assert np.array_equal(z['equality_rhs'],arr['equality_rhs']) and np.count_nonzero(z['equality_rhs'])==0
  assert np.array_equal(z['native_bounds'],arr['bounds'][:-1]) and np.array_equal(z['native_cost'],arr['cost'][:-1]) and np.count_nonzero(z['native_cost'])==0
  assert np.all(np.isneginf(z['native_bounds'][:,0])) and np.all(np.isposinf(z['native_bounds'][:,1]))
  L,U=sorted([e*F(float(q)) for q in arr['bounds'][-1]]);assert L<U and F(float(z['lower_v']))==L and F(float(z['upper_v']))==U and F(float(z['objective_sign']))==e
  assert np.array_equal(z['original_scalar_bounds'],arr['bounds'][-1])
  ids=np.load(src.parent/'row_ids.npy');assert np.array_equal(ids,z['source_row_ids'])
  positions=np.concatenate([z['hard_source_positions'],z['coupled_source_positions']]);assert np.array_equal(np.sort(positions),np.arange(A.shape[0]))
  verified=np.zeros(A.shape[0],dtype=np.uint8);scalar_count=0;native_count=0
  for kind,M in [('hard',H),('coupled',S)]:
   pos=z[kind+'_source_positions'];assert M.shape==(len(pos),n) and np.array_equal(z[kind+'_original_row_ids'],ids[pos]) and np.array_equal(z[kind+'_rhs'],arr['rhs'][pos])
   assert M.has_sorted_indices and M.has_canonical_format
   for j,i in enumerate(pos):
    ac=A.indices[A.indptr[i]:A.indptr[i+1]];av=A.data[A.indptr[i]:A.indptr[i+1]];keep=ac<n
    jc=M.indices[M.indptr[j]:M.indptr[j+1]];jv=M.data[M.indptr[j]:M.indptr[j+1]]
    assert np.array_equal(ac[keep],jc) and np.array_equal(av[keep],jv),(role,int(i))
    last=av[~keep]
    if kind=='hard':assert len(last)==0
    else:assert len(last)==1 and F(float(last[0]))==-e;scalar_count+=1
    native_count+=len(jv);assert verified[i]==0;verified[i]=1
  assert np.all(verified==1) and native_count+scalar_count==A.nnz
  np.save(out/(role+'_row_copy_verified.npy'),verified)
  systems[role]=dict(rows_verified=len(verified),hard_rows=H.shape[0],coupled_residuals=S.shape[0],native_directions=n,native_coefficients_verified=native_count,scalar_signs_verified=scalar_count,coefficient_copy_errors=0,rhs_copy_errors=0,native_bounds_cost_equalities_exact=True,equality_rows=1366,lower_v=str(L),upper_v=str(U),all_rows_retained_in_functional_form=True,proof_517_inherited_not_repeated=True)
 # Prove saved proximal minimizers by first-order KKT, without evaluating the producer's threshold algorithm.
 examples=read(prep/'small_exact_examples.json');prox_checks=[]
 for c in examples['proximal_examples']:
  w,b,sigma,q,lam=[list(map(F,c[k])) for k in ['w','b','sigma','q','solution']];L,U=F(c['L']),F(c['U']);a=F(c['threshold']);s=sum(lam)
  assert len(w)==len(b)==len(sigma)==len(q)==len(lam) and U>L and all(t>0 for t in sigma)
  assert all(x>=0 for x in lam) and 0<=a<=U-L and F(c['mass'])==s
  if s<1:assert a==0
  if s>1:assert a==U-L
  derivatives=[]
  for i in range(len(w)):
   assert q[i]==w[i]-sigma[i]*(b[i]+L)
   grad=(lam[i]-w[i])/sigma[i]+b[i]+L+a
   assert (grad==0 if lam[i]>0 else grad>=0);derivatives.append(str(grad))
  quadratic=sum((x-y)**2/(2*t) for x,y,t in zip(lam,w,sigma));conj=L*(s-1)+(U-L)*max(F(0),s-1)
  objective=quadratic+sum(bi*xi for bi,xi in zip(b,lam))+conj
  prox_checks.append(dict(name=c['name'],exact_kkt=True,strict_convexity_unique_minimizer=True,mass=str(s),threshold=str(a),zero_bound_derivatives=derivatives,objective=str(objective),optimizer_calls=0))
 assert len(prox_checks)==6
 fen_checks=[]
 for c in examples['fenchel_and_bound_lift_examples']:
  e=F(c['e']);L,U=F(c['L']),F(c['U']);r,lam=[list(map(F,c[k])) for k in ['r','lam']];s=sum(lam);g=F(c['g']);gs=F(c['conjugate']);alpha,beta=F(c['alpha']),F(c['beta'])
  assert max(r)<=U and g==max([L]+r) and all(x>=0 for x in lam)
  # Independent piecewise conjugate values and exact Fenchel equality.
  assert gs==(L*(s-1) if s<=1 else U*(s-1))
  assert sum(x*y for x,y in zip(r,lam))==g+gs
  assert alpha>=0 and beta>=0 and 1-s+alpha-beta==0 and U*alpha-L*beta==gs
  assert alpha*(U-g)==0 and beta*(g-L)==0 and all(x*(g-y)==0 for x,y in zip(lam,r))
  tu,tl=F(c['original_upper_multiplier']),F(c['original_lower_multiplier']);t=F(c['original_t']);lo,hi=sorted([e*L,e*U])
  assert t==e*g and lo<=t<=hi and e-e*s+tu-tl==0 and tu*(hi-t)==0 and tl*(t-lo)==0
  fen_checks.append(dict(name=c['name'],exact_primal_conjugate_fenchel_equality=True,original_scalar_stationarity_and_complementarity=True))
 assert len(fen_checks)==3
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,systems=systems,new_small_proximal_kkt_checks=prox_checks,new_small_fenchel_dual_checks=fen_checks,prior_517_proof_inherited=True,production_imported=False,old_fixture_repeated=False,full_size_proximal_updates=0,lp_calls=0,linear_solves=0,model_calls=0,root_calls=0,finite_maps=0,new_solver=False,new_candidate=False,physical_admission=False,limits='Source specialization/copies and six local exact proximal minimizers/three Fenchel-lift identities verified. General conjugacy derives from explicit supremum, not sampled physical states. No numerical whole-path method, convergence rate, feasibility or speed result.')
 (out/'independent_audit.json').write_text(json.dumps(r,indent=2));(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()},indent=2));print(json.dumps(r,indent=2),flush=True)
if __name__=='__main__':main()
