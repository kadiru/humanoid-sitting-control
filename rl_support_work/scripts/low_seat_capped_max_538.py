"""Exact saved-original epigraph elimination and capped-maximum conjugacy; no LP solver."""
from pathlib import Path
from fractions import Fraction as F
import hashlib,json
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def fs(q):return str(q.numerator)+'/'+str(q.denominator)
SPEC={
 'scope':'Preparation only: exact-real epigraph elimination in both immutable516 ORIGINAL representative systems. All78790 native direction variables INCLUDING21840 auxiliary torques remain. No522 projection or504 mixed transform is used. No LP, optimizer, candidate or full-size proximal update.',
 'source_proof':'Inherit517 REAL original-system implication proof after named hashes; do not regroup or reprove. Failed516 synthetic fixture and staged recovery stay historical. Full490 original checks remain mandatory.',
 'canonicalization':'Let e=cost[last] in{+1,-1}, v=e*t, L=min(e*lower_t,e*upper_t), U=max(...). Every coupled row must have coefficient -e, every hard row zero. Then Hx<=bH, Sx-v<=bS, L<=v<=U, min v. Native costs zero; native bounds and equality rows/RHS retained unchanged.',
 'primal_function':'For r=Sx-bS, g(r)=max(L,max(r)) if max(r)<=U, else +infinity. Eliminate v by minimizing its feasible interval[max(L,max(r)),U]. Reconstruct v=g(r),t=e*v, with native x unchanged. No row is redundant or removed: hard rows and ALL coupled residuals remain. Upper domain must never be omitted.',
 'conjugate':'g*(lambda)=L*(sum(lambda)-1)+(U-L)*max(sum(lambda)-1,0) for lambda>=0, +infinity otherwise. For nonnegative lambda, supremum reduces to sup_(a<=U) sum(lambda)*a-max(L,a); for any negative component, send that r component to -infinity. This proves the stated conjugate exactly.',
 'saddle':'min_x max_(yH>=0,lambda>=0) yH.(Hx-bH)+lambda.(Sx-bS)-g*(lambda), with unchanged native domain. This is a reformulation, not a complete numerical algorithm or convergence/performance certificate.',
 'dual_lift':'For s=sum(lambda), canonical v upper/lower multipliers alpha=max(s-1,0), beta=max(1-s,0) give1-s+alpha-beta=0. Dual objective -bH.yH-bS.lambda-U*alpha+L*beta equals -b.y-g*. If e=-1, swap bound multipliers for original t. Native stationarity H.T*yH+S.T*lambda (plus any unchanged native-domain dual terms) must still hold.',
 'weighted_prox':'For sigma>0, prox of b.lambda+g* in diagonal metric1/sigma: q=w-sigma*(b+L), width=U-L; lambda_i=max(q_i-sigma_i*a,0). a=0 if sum max(q,0)<=1; a=width if sum max(q-sigma*width,0)>=1; otherwise solve sum max(q-sigma*a,0)=1 by exact sorted active breakpoints q/sigma. Only NEW small exact arithmetic examples are executed; not a robot solver update.',
 'future_certification':'Any numerical method must specify complete initialization, step sizes, residual checks, resource caps and independent audit before launch. Native returned coordinates cannot be repaired. Complete original t direction must pass source and FULL490 inequality/equality/bound checks at unchanged1e-7. Actual primary result sets interior ceiling, exact unscaled interval eta>0 precedes immutable490 encoding. No522 torque midpoint lift is involved in this native-variable formulation. All actual unit/geometry/filter/runtime/physical/PPO gates remain.',
 'limits':'One dense scalar column eliminated analytically, not a smaller feasible constraint set. Stored native direction count78790 is21840 larger than the torque-projected56950 route. No speedup, conditioning gain, deadline explanation, global infeasibility, actual path, force or effort claim.',
 'resources':'ONE preparation of two saved systems; no iterative optimizer. Six new small rational proximal examples and three exact primal/conjugate/dual-bound examples. Independent audit copies/support and rational KKT only, no optimizer/factor/linear solve. No old fixture replay.',
 'reference':'General proximal definition and uniqueness: Parikh and Boyd, Proximal Algorithms, https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf . Capped-maximum formulas and source specialization derived explicitly here.'}
def split(A,arr,source_ids,out):
 assert A.format=='csr' and A.has_sorted_indices and A.has_canonical_format
 n=A.shape[1]-1;e=float(arr['cost'][-1]);assert e in(-1.,1.) and not np.any(arr['cost'][:-1])
 assert np.all(np.isneginf(arr['bounds'][:-1,0])) and np.all(np.isposinf(arr['bounds'][:-1,1]))
 col=A[:,n].toarray().ravel();assert np.all((col==0)|(col==-e))
 hard=np.flatnonzero(col==0);coupled=np.flatnonzero(col==-e);assert len(coupled)>0
 lo,hi=arr['bounds'][-1];L,U=sorted([e*float(lo),e*float(hi)]);assert np.isfinite(L+U) and L<U
 H=A[hard,:n].tocsr();S=A[coupled,:n].tocsr()
 sparse.save_npz(out/'hard.npz',H);sparse.save_npz(out/'coupled.npz',S)
 np.savez(out/'arrays.npz',hard_rhs=arr['rhs'][hard],coupled_rhs=arr['rhs'][coupled],native_cost=arr['cost'][:-1],native_bounds=arr['bounds'][:-1],equality_rhs=arr['equality_rhs'],source_row_ids=source_ids,hard_source_positions=hard,coupled_source_positions=coupled,hard_original_row_ids=source_ids[hard],coupled_original_row_ids=source_ids[coupled],objective_sign=np.array(e),lower_v=np.array(L),upper_v=np.array(U),original_scalar_bounds=arr['bounds'][-1])
 return dict(source_shape=list(A.shape),source_nonzeros=int(A.nnz),native_directions=n,hard_rows=len(hard),coupled_residuals=len(coupled),hard_nonzeros=int(H.nnz),coupled_native_nonzeros=int(S.nnz),native_nonzeros=int(H.nnz+S.nnz),scalar_coefficients_analytically_removed=len(coupled),constraint_rows_omitted=0,native_variables_eliminated=0,objective_sign=e,lower_v=L,upper_v=U,all_native_bounds_unbounded=True,all_native_costs_zero=True)
def prox(w,b,sigma,L,U):
 q=[wi-si*(bi+L) for wi,bi,si in zip(w,b,sigma)];width=U-L
 assert width>0 and all(si>0 for si in sigma)
 mass=lambda a:sum(max(F(0),qi-si*a) for qi,si in zip(q,sigma))
 if mass(F(0))<=1:a=F(0);branch='lower_hinge'
 elif mass(width)>=1:a=width;branch='upper_hinge'
 else:
  order=sorted(range(len(q)),key=lambda i:(-q[i]/sigma[i],i));qs=ss=F(0);a=None
  for k,i in enumerate(order):
   qs+=q[i];ss+=sigma[i];z=(qs-1)/ss
   upper=q[i]/sigma[i];lower=q[order[k+1]]/sigma[order[k+1]] if k+1<len(q) else None
   if z<=upper and (lower is None or z>=lower):a=z;break
  assert a is not None and 0<a<width and mass(a)==1;branch='simplex_hinge'
 lam=[max(F(0),qi-si*a) for qi,si in zip(q,sigma)]
 return dict(branch=branch,threshold=fs(a),q=[fs(x) for x in q],solution=[fs(x) for x in lam],mass=fs(sum(lam)))
def examples():
 # These fixtures are new, local, exact proximal arithmetic; no LP routine is called.
 cases=[
  ('primary_inactive',[F(1,4),0,-1],[0,0,0],[1,2,1],0,1),
  ('primary_simplex',[1,1,0],[0,0,0],[1,2,1],0,1),
  ('primary_upper',[3,2,-1],[0,0,0],[1,1,2],0,1),
  ('witness_nonuniform',[F(1,4),F(-1,4),F(-3,4)],[F(1,4),F(-1,2),F(1,3)],[2,1,3],F(-1,2),0),
  ('boundary_mass_one',[1,0,-1],[0,0,0],[1,2,1],0,1),
  ('zero_output',[-1,-2,0],[0,0,0],[1,1,1],0,1)]
 out=[]
 for name,w,b,s,L,U in cases:
  w,b,s=[list(map(F,z)) for z in(w,b,s)];L,U=F(L),F(U)
  out.append(dict(name=name,w=list(map(fs,w)),b=list(map(fs,b)),sigma=list(map(fs,s)),L=fs(L),U=fs(U),**prox(w,b,s,L,U)))
 # Fenchel equality at all three branches and canonical-to-original scalar dual lift.
 fen=[]
 for name,e,L,U,r,lam in [('lower',1,F(0),F(1),[0,F(-1,2)],[F(1,4),0]),('interior',1,F(0),F(1),[F(1,3),F(1,3)],[F(1,4),F(3,4)]),('upper_witness',-1,F(-1,2),F(0),[0,F(-1,3)],[2,0])]:
  r=list(map(F,r));lam=list(map(F,lam));s=sum(lam);g=max([L]+r);alpha=max(F(0),s-1);beta=max(F(0),1-s);conj=L*(s-1)+(U-L)*max(F(0),s-1)
  assert max(r)<=U and sum(x*y for x,y in zip(r,lam))==g+conj
  tu,tl=(alpha,beta) if e==1 else(beta,alpha)
  assert e-e*s+tu-tl==0
  fen.append(dict(name=name,e=e,L=fs(L),U=fs(U),r=list(map(fs,r)),lam=list(map(fs,lam)),g=fs(g),conjugate=fs(conj),alpha=fs(alpha),beta=fs(beta),original_t=fs(e*g),original_upper_multiplier=fs(tu),original_lower_multiplier=fs(tl)))
 return dict(proximal_examples=out,fenchel_and_bound_lift_examples=fen,lp_calls=0,full_size_updates=0,solver_candidates=0)
