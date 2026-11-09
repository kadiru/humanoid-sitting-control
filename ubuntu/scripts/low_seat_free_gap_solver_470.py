"""New JOINT free-gap collocation formulation; no old optimizer is continued.
The finite normal-drop chart468 preserves exact endpoint sum, while chart460
preserves bounded total-normal loads. Dependent f remains fully constrained.
All LP derivatives are recentered CONTINUOUS pre-round chart derivatives;
actual mapped/rounded/retracted rows decide acceptance. No physical admission.
"""
import json
from fractions import Fraction as F
import numpy as np
from scipy import sparse
import low_seat_collocation_solver_448 as base
from low_seat_gap_coordinates_v2_467 import implicit,DEP
from low_seat_exact_gap_chart_468 import load_domain,initial,map_drops,gaps,pair,unpack,fq
from low_seat_simplex_retraction_460 import decode_group
D=base.D;sha=base.sha;read=base.read;values=base.values;summary=base.summary
SPEC=dict(base.SPEC)
SPEC.update(max_predictors=24,max_corrections_per_predictor=4,max_linear_programs=120,max_consecutive_load_roundings=120,seed='Original440 plus newly encoded466/468 gap targets; first complete lift is NEW and may be infeasible.',budget_justification='18 half-reductions from source440 merit to unchanged2^-20 guard plus6 bounded rejection opportunities; at most4 corrections each. This is a separate free-gap formulation, never an extension of462.',coordinate_domain='Original11906 f/p/load +/- .02 box around440 unchanged; dependent f bounds explicit. New206 normal-drop bounds retain all original caps plus prepared root/roundoff guard. No extra gap or allocation sweep.',normal_drop_map='Audited468 exact residual pivot with bounded joint rounding; no endpoint-sum accumulation.',equality_policy='All original normal/hover primal rows still checked. Their analytic zero tangent rows impose homogeneous conservation, not restoration of admissible floating residuals.',physical_admission=False)

def load_saved():
 d=base.load_saved();P=D/'gap_coordinate_preparation_v3_466';p=dict(np.load(P/'implicit_map.npz'));d.update(gap_domain=load_domain(),gap_layout=read(P/'layout.json'),B=sparse.load_npz(P/'coordinate_basis.npz').tocsr(),C=sparse.load_npz(P/'original_control_box_matrix.npz').tocsr(),initial_target=p['target_controls'].copy())
 cols=d['gap_layout']['columns'];n=len(cols);K=len(d['initialX']);nr=[];nc=[];nv=[];ndcols=[];groups=[];by={}
 for j,c in enumerate(cols):
  if c['kind']=='normal_drop':ndcols.append(j);nr.extend([c['edge'],c['pivot']]);nc.extend([j,j]);nv.extend([1.,-1.])
  else:by.setdefault((c['state'],c['kind']),[]).append((j,c))
 for (b,kind),entries in by.items():
  groups.append(dict(state=b,kind=kind,columns=[v[0] for v in entries],entries=[v[1] for v in entries]))
 d.update(N=sparse.csr_matrix((nv,(nr,nc)),shape=(206,n)),normal_columns=np.asarray(ndcols),map_groups=groups)
 return d

def linearization(d,J):
 T,G,_,_,_=implicit(J);GB=sparse.block_diag(list(G),format='csr')@d['B'];A=(d['M']@GB).multiply((1/d['a']['scale'])[:,None]).tocsr();A.eliminate_zeros();E=(d['E']@GB).tocsr();E.eliminate_zeros();assert E.nnz==0;C=(d['C']@GB).tocsr()
 return T,G,A,E,C

def assemble(d,X,Y,J,target,drops,trust,mode='predictor',merit_ceiling=None,origin=None):
 _,r,e=values(d,X,Y);T,G,A,E,C=linearization(d,J);keep=d['a']['keep'];soft=d['a']['soft']&keep;hard=keep&~soft;z=d['C']@np.c_[X,Y].ravel();n=A.shape[1];radius=SPEC['coordinate_domain_radius']*trust
 assert np.max(abs(e[:-1]))<=1e-13
 variable=np.diff(A.indptr)>0;guard=np.zeros(len(r));guard[soft&variable]=SPEC['predictor_guard'];hp=hard&variable;guard[hp]=np.where(r[hp]>0,np.minimum(SPEC['predictor_guard'],.5*r[hp]),SPEC['predictor_guard'])
 if mode=='predictor':
  dl=np.maximum(d['lower']-z,-radius);du=np.minimum(d['upper']-z,radius);top=sparse.hstack([-A[keep],sparse.csr_matrix(-soft[keep,None].astype(float))],format='csr');rhs=r[keep]-guard[keep];last=[0.,max(1.,2*max(0.,float(-r[soft].min())))];tail=[];tailrhs=[]
 else:
  assert origin is not None and merit_ceiling is not None
  dl=np.maximum.reduce([d['lower']-z,origin-radius-z,np.full(len(z),-.25*radius)]);du=np.minimum.reduce([d['upper']-z,origin+radius-z,np.full(len(z),.25*radius)])
  top=sparse.hstack([-A[keep],sparse.csr_matrix((keep.sum(),1))],format='csr');rhs=r[keep]-guard[keep]+soft[keep]*merit_ceiling;last=[0.,.25]
  tail=[sparse.hstack([C,sparse.csr_matrix(-np.ones((len(z),1))*radius)]),sparse.hstack([-C,sparse.csr_matrix(-np.ones((len(z),1))*radius)])];tailrhs=[np.zeros(len(z)),np.zeros(len(z))]
 cb=sparse.hstack([C,sparse.csr_matrix((len(z),1))],format='csr');N=d['N'];nb=sparse.hstack([N,sparse.csr_matrix((206,1))],format='csr');dom=d['gap_domain'];ndlo=np.array([float(l-v) for l,v in zip(dom['lower'],drops)]);ndhi=np.array([float(u-v) for u,v in zip(dom['upper'],drops)])
 mat=sparse.vstack([top]+tail+[cb,-cb,nb,-nb],format='csr');rhs=np.concatenate([rhs]+tailrhs+[du,-dl,ndhi,-ndlo]);eq=sparse.csr_matrix((int(d['ea']['optimizer_keep'].sum()),n+1));bounds=np.c_[np.r_[np.full(n,-np.inf),last[0]],np.r_[np.full(n,np.inf),last[1]]]
 return dict(X=X,Y=Y,J=J,target=target,drops=drops,T=T,G=G,A=A,E=E,C=C,r=r,e=e,z=z,guard=guard,matrix=mat,rhs=rhs,equality=eq,equality_rhs=np.zeros(eq.shape[0]),bounds=bounds,cost=np.r_[np.zeros(n),1.],mode=mode,merit_ceiling=merit_ceiling,full_step_lower=dl,full_step_upper=du,normal_step_lower=ndlo,normal_step_upper=ndhi,trust=trust,origin=z if origin is None else origin)

def save_assembly(a,folder):
 folder.mkdir(exist_ok=False)
 for key in ['matrix','equality','A','E','C']:sparse.save_npz(folder/(key+'.npz'),a[key])
 np.savez_compressed(folder/'input.npz',**{k:a[k] for k in ['X','Y','J','target','r','e','guard','rhs','equality_rhs','bounds','cost','z','full_step_lower','full_step_upper','normal_step_lower','normal_step_upper','origin']})
 (folder/'metadata.json').write_text(json.dumps(dict(mode=a['mode'],merit_ceiling=a['merit_ceiling'],trust=a['trust'],drops=[pair(x) for x in a['drops']])))

# No LP is called by preparation or this module on import.
