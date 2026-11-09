"""ONE complete-clock algorithm using the new conserved bounded chart460.
No456 trial/450 final is continued. All controls are joint, including phases.
Normal equations are conservation invariants with the audited rounding bound;
their original primal tolerances remain mandatory. All pivot bounds remain.
"""
import json
import numpy as np
from scipy import sparse
import low_seat_collocation_solver_448 as base
from low_seat_simplex_retraction_460 import decode,CONTRACT,D,sha
load_saved=base.load_saved;values=base.values;summary=base.summary;read=base.read
SPEC=dict(base.SPEC)
SPEC.update(max_predictors=24,max_corrections_per_predictor=4,max_linear_programs=120,
 max_consecutive_map_roundings=120,coordinate_retraction='exact rational common-alpha bounded load null-space chart460',
 seed='immutable440, no450/456 final or rejected trial',
 budget_justification='18 half-reductions cover original440 rate merit to unchanged2^-20 step guard; six rejection opportunities. Four correctors bound kinematic/gap defects.120 maps also satisfies the independently audited normal-rounding contract. This is one new algorithm; no extension.',
 normal_equation_policy='All original normal rows checked at every path; zero tangent normal rows impose homogeneous conservation, never correction of their tiny already-admissible rounding residual.',
 physical_admission=False)

def chart():
 P=D/'simplex_preparation_460'
 return read(P/'layout.json'),sparse.load_npz(P/'null_basis.npz').tocsr()

def assemble(d,R,X,Y,J,trust,mode='predictor',merit_ceiling=None,origin=None):
 a=base.assemble(d,X,Y,J,trust,mode,merit_ceiling,origin)
 if np.max(abs(a['e'][:-1]))>1e-13:raise ValueError('normal invariant already outside unchanged bound')
 transform=sparse.block_diag([R,sparse.eye(1)],format='csr')
 oldmat=a['matrix'];oldeq=a['equality'];oldrhs=a['equality_rhs'];oldbounds=a['bounds'];n=R.shape[1]
 eq=(oldeq@transform).tocsr();kept_ids=np.flatnonzero(d['ea']['optimizer_keep']);normal=kept_ids<len(a['e'])-1
 normalmat=eq[normal].copy();normalmat.eliminate_zeros();assert normalmat.nnz==0
 erhs=oldrhs.copy();erhs[normal]=0. # Homogeneous tangent conservation, not a primal waiver.
 boundmat=sparse.hstack([R,sparse.csr_matrix((R.shape[0],1))],format='csr')
 mat=sparse.vstack([oldmat@transform,boundmat,-boundmat],format='csr')
 rhs=np.r_[a['rhs'],oldbounds[:-1,1],-oldbounds[:-1,0]]
 bounds=np.c_[np.r_[np.full(n,-np.inf),oldbounds[-1,0]],np.r_[np.full(n,np.inf),oldbounds[-1,1]]]
 a.update(original_matrix=oldmat,original_equality=oldeq,original_equality_rhs=oldrhs,original_bounds=oldbounds,original_rhs=a['rhs'],matrix=mat,rhs=rhs,equality=eq,equality_rhs=erhs,bounds=bounds,cost=np.r_[np.zeros(n),1.],normal_conservation_rows=normal)
 return a

def solve_lp(a,folder):
 result,meta=base.solve_lp(a,folder)
 sparse.save_npz(folder/'original_matrix.npz',a['original_matrix']);sparse.save_npz(folder/'original_equality.npz',a['original_equality'])
 np.savez_compressed(folder/'conversion.npz',original_rhs=a['original_rhs'],original_equality_rhs=a['original_equality_rhs'],original_bounds=a['original_bounds'],normal_conservation_rows=a['normal_conservation_rows'])
 return result,meta

def map_interval(d,a,trust,origin):
 radius=SPEC['coordinate_domain_radius']*trust;lo=np.maximum(d['lower'],origin-radius);hi=np.minimum(d['upper'],origin+radius)
 if a['mode']=='correction':
  lo=np.maximum(lo,a['z']-SPEC['correction_radius_fraction']*radius);hi=np.minimum(hi,a['z']+SPEC['correction_radius_fraction']*radius)
 return lo,hi

class SimplexRetraction(base.Retraction):
 def __init__(self,d,groups):super().__init__(d);self.groups=groups
 def run_simplex(self,a,tangent,lower,upper):
  origin=a['z'];controls,alphas,sum_errors=decode(origin,tangent,lower,upper,self.groups)
  d=self.d;self.counts['trial_paths']+=1;assert self.counts['trial_paths']<=CONTRACT['max_consecutive_map_roundings']
  dc=np.zeros(d['c']['reference'].size);dc[d['free']]=controls-origin;dc=dc.reshape(-1,10)
  x=a['X']+np.einsum('nic,nc->ni',a['T'],dc)
  allcontrols=a['X'][:,d['c']['control']].copy().ravel();allcontrols[d['free']]=controls
  x[:,d['c']['control']]=allcontrols.reshape(-1,10);x[d['g']['fixed']]=d['initialX'][d['g']['fixed']]
  dep=d['c']['dependent'];history=[];mapped=dict(mapped_controls=controls,alphas=alphas,sum_errors=sum_errors)
  for it in range(SPEC['root_max_updates']+1):
   y=self.val(x);res=y[:,:14];bad=(np.max(abs(res),axis=1)>SPEC['root_absolute_tolerance'])&~d['g']['fixed'];history.append(float(abs(res).max()))
   if not np.any(bad):return dict(passed=True,X=x,Y=y,root_history=history,**mapped)
   if it==SPEC['root_max_updates']:break
   j=self.jac(x)
   try:step=np.linalg.solve(j[:,:14][:,:,dep],-res[...,None])[...,0]
   except np.linalg.LinAlgError:return dict(passed=False,reason='singular root Jacobian',X=x,Y=y,root_history=history,**mapped)
   norm=np.linalg.norm(res,axis=1);accepted=~bad
   for h in range(SPEC['root_max_halvings']):
    active=np.flatnonzero(~accepted);candidate=x.copy();candidate[active[:,None],dep]+=step[active]*2.**(-h);v=self.val(candidate);take=(~accepted)&(np.linalg.norm(v[:,:14],axis=1)<norm);x[take]=candidate[take];accepted|=take
    if accepted.all():break
   self.counts['root_updates']+=1
   if not accepted.all():return dict(passed=False,reason='root line search cap',X=x,Y=self.val(x),root_history=history,**mapped)
  return dict(passed=False,reason='root update cap',X=x,Y=y,root_history=history,**mapped)
