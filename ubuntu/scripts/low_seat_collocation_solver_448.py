"""Bounded sparse whole-clock feasibility algorithm, NEW domain444/446v2.
Saved440 is an infeasible research seed. No430/424 optimizer is continued.
All phase/load controls move jointly; the14 foot/moment coordinates retract
with controls fixed. No inverse-gap root, target/force clipping, time snapping,
allocation search, physical solver change or actual command execution.
"""
from pathlib import Path
import json,time,datetime
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from low_seat_clock_transcription_444 import D,Mechanics,load_inputs,sha
P=D/'clock_transcription_preparation_444';Q=D/'clock_row_preparation_446';V=D/'clock_row_preparation_v2_446'
read=lambda p:json.loads(Path(p).read_text())
SPEC=dict(max_predictors=6,max_corrections_per_predictor=2,max_linear_programs=18,lp_time_limit_s=45.,lp_method='highs',lp_primal_dual_tolerance=1e-9,independent_linear_bound=1e-7,coordinate_domain_radius=.02,initial_trust_fraction=1.,minimum_trust_fraction=1/32,root_max_updates=10,root_max_halvings=9,root_absolute_tolerance=1e-13,hover_numerical_bound=1e-8,strict_inequality_bound=0.,predictor_guard=2.**-20,acceptance_ratio=.1,correction_radius_fraction=.25)

def load_saved():
 a=dict(np.load(V/'row_arrays.npz'));g=dict(np.load(P/'mesh.npz'));c=dict(np.load(Q/'coordinate_map.npz'));ea=dict(np.load(Q/'equality_arrays.npz'))
 M=sparse.load_npz(V/'all_rows_matrix.npz');E=sparse.load_npz(Q/'equality_matrix.npz');free=c['free'];ref=c['reference'].ravel()[free]
 lower=np.maximum(c['lower'].ravel()[free],ref-SPEC['coordinate_domain_radius']);upper=np.minimum(c['upper'].ravel()[free],ref+SPEC['coordinate_domain_radius'])
 return dict(a=a,g=g,c=c,ea=ea,M=M,E=E,free=free,reference=ref,lower=lower,upper=upper,initialX=a['features'][:,:24].copy(),initialY=a['features'][:,24:].copy(),initialJ=g['canonical_J'].copy())

def values(d,X,Y):
 features=np.c_[X,Y];r=(d['M']@features.ravel()+d['a']['constant'])/d['a']['scale'];e=d['E']@features.ravel()+d['ea']['constant']
 return features,r,e

def linearization(d,J):
 dep=d['c']['dependent'];control=d['c']['control'];K=len(J);T=np.zeros((K,24,10));T[:,control,:]=np.eye(10);T[:,dep,:]=-np.linalg.solve(J[:,:14][:,:,dep],J[:,:14][:,:,control]);G=np.concatenate([T,np.einsum('noi,nic->noc',J,T)],axis=1);B=sparse.block_diag(list(G),format='csr')[:,d['free']]
 A=(d['M']@B).multiply((1/d['a']['scale'])[:,None]).tocsr();E=(d['E']@B).tocsr()
 return T,G,A,E

def summary(d,X,Y,r,e):
 keep=d['a']['keep'];soft=d['a']['soft']&keep;hard=keep&~soft;phase=d['a']['groups']=='mesh_phase';z=X[:,d['c']['control']].ravel()[d['free']]
 return dict(minimum_retained_buffered_row=float(r[keep].min()),minimum_hard_row=float(r[hard].min()),minimum_soft_row=float(r[soft].min()),soft_merit=max(0.,float(-r[soft].min())),hard_pass=bool(np.all(r[hard]>=0)),all_inequalities_pass=bool(np.all(r[keep]>=0)),minimum_phase=float(r[phase].min()),monotone_phase_pass=bool(np.all(r[phase]>=0)),coordinate_bound_minimum=float(min((z-d['lower']).min(),(d['upper']-z).min())),maximum_retained_equality=float(abs(e[d['ea']['optimizer_keep']]).max()),maximum_kinematic_residual=float(abs(Y[:,:14]).max()),minimum_corner_weight=float(X[:,16:24].min()),fixed_states_exact=bool(np.array_equal(X[d['g']['fixed']],d['initialX'][d['g']['fixed']])),inactive_exact_zero=bool(np.all(X[:,16:24][d['g']['inactive']]==0)))

def assemble(d,X,Y,J,trust,mode='predictor',merit_ceiling=None,origin=None):
 features,r,e=values(d,X,Y);T,G,A,E=linearization(d,J);keep=d['a']['keep'];soft=d['a']['soft']&keep;hard=keep&~soft;z=X[:,d['c']['control']].ravel()[d['free']]
 # All previously retained rows stay present. EXACT zero Jacobian rows get no
 #extra predictor guard; no small-Jacobian row is removed or thresholded.
 variable=np.diff(A.indptr)>0;guard=np.zeros(len(r));guard[soft&variable]=SPEC['predictor_guard'];hp=hard&variable
 guard[hp]=np.where(r[hp]>0,np.minimum(SPEC['predictor_guard'],.5*r[hp]),SPEC['predictor_guard'])
 radius=SPEC['coordinate_domain_radius']*trust
 if mode=='predictor':
  dl=np.maximum(d['lower']-z,-radius);du=np.minimum(d['upper']-z,radius);cost=np.r_[np.zeros(len(z)),1.];scol=-soft[keep].astype(float)
  mat=sparse.hstack([-A[keep],sparse.csr_matrix(scol[:,None])],format='csr');rhs=r[keep]-guard[keep];bounds=np.c_[np.r_[dl,0.],np.r_[du,max(1.,2*max(0.,float(-r[soft].min())))]]
 else:
  assert origin is not None and merit_ceiling is not None
  dl=np.maximum.reduce([d['lower']-z,origin-radius-z,np.full(len(z),-SPEC['correction_radius_fraction']*radius)])
  du=np.minimum.reduce([d['upper']-z,origin+radius-z,np.full(len(z),SPEC['correction_radius_fraction']*radius)])
  # Correction minimizes normalized displacement, with the predictor's rate
  #improvement fixed as a constraint. No hard-feasibility epigraph is introduced.
  cost=np.r_[np.zeros(len(z)),1.];top=sparse.hstack([-A[keep],sparse.csr_matrix((int(keep.sum()),1))],format='csr');rhs=r[keep]-guard[keep]+soft[keep]*merit_ceiling
  I=sparse.eye(len(z),format='csr');tail=sparse.vstack([sparse.hstack([I,sparse.csr_matrix(-np.ones((len(z),1))*radius)]),sparse.hstack([-I,sparse.csr_matrix(-np.ones((len(z),1))*radius)])],format='csr');mat=sparse.vstack([top,tail],format='csr');rhs=np.r_[rhs,np.zeros(2*len(z))];bounds=np.c_[np.r_[dl,0.],np.r_[du,SPEC['correction_radius_fraction']]]
 eq=sparse.hstack([E[d['ea']['optimizer_keep']],sparse.csr_matrix((int(d['ea']['optimizer_keep'].sum()),1))],format='csr');erhs=-e[d['ea']['optimizer_keep']]
 return dict(X=X,Y=Y,J=J,T=T,G=G,A=A,E=E,r=r,e=e,z=z,guard=guard,matrix=mat,rhs=rhs,equality=eq,equality_rhs=erhs,bounds=bounds,cost=cost,mode=mode,merit_ceiling=merit_ceiling)

class Retraction:
 def __init__(self,d):
  self.d=d;self.oracle=Mechanics(load_inputs(),d['g']['canonical']);self.counts=dict(value_requests=0,jacobian_requests=0,trial_paths=0,root_updates=0)
 def val(self,x):self.counts['value_requests']+=1;return self.oracle.values(x)
 def jac(self,x):self.counts['jacobian_requests']+=1;return self.oracle.jacobian(x)
 def run(self,X,T,delta):
  d=self.d;self.counts['trial_paths']+=1;dc=np.zeros(d['c']['reference'].size);dc[d['free']]=delta;dc=dc.reshape(-1,10);x=X+np.einsum('nic,nc->ni',T,dc);x[:,d['c']['control']]=X[:,d['c']['control']]+dc;x[d['g']['fixed']]=d['initialX'][d['g']['fixed']];dep=d['c']['dependent'];history=[]
  for it in range(SPEC['root_max_updates']+1):
   y=self.val(x);res=y[:,:14];bad=(np.max(abs(res),axis=1)>SPEC['root_absolute_tolerance'])&~d['g']['fixed'];history.append(float(abs(res).max()))
   if not np.any(bad):return dict(passed=True,X=x,Y=y,root_history=history)
   if it==SPEC['root_max_updates']:break
   j=self.jac(x)
   try:step=np.linalg.solve(j[:,:14][:,:,dep],-res[...,None])[...,0]
   except np.linalg.LinAlgError:return dict(passed=False,reason='singular root Jacobian',X=x,Y=y,root_history=history)
   norm=np.linalg.norm(res,axis=1);accepted=~bad
   for h in range(SPEC['root_max_halvings']):
    active=np.flatnonzero(~accepted);candidate=x.copy();candidate[active[:,None],dep]+=step[active]*2.**(-h);v=self.val(candidate);take=(~accepted)&(np.linalg.norm(v[:,:14],axis=1)<norm);x[take]=candidate[take];accepted|=take
    if accepted.all():break
   self.counts['root_updates']+=1
   if not accepted.all():return dict(passed=False,reason='root line search cap',X=x,Y=self.val(x),root_history=history)
  return dict(passed=False,reason='root update cap',X=x,Y=y,root_history=history)

def solve_lp(a,folder):
 folder.mkdir(exist_ok=False);sparse.save_npz(folder/'matrix.npz',a['matrix']);sparse.save_npz(folder/'equality.npz',a['equality']);np.savez_compressed(folder/'input.npz',rhs=a['rhs'],equality_rhs=a['equality_rhs'],bounds=a['bounds'],cost=a['cost'],X=a['X'],Y=a['Y'],guard=a['guard']);start=time.monotonic()
 if np.any(a['bounds'][:,0]>a['bounds'][:,1]):res=None;meta=dict(status=-10,message='empty bounded correction box',candidate=False)
 else:
  res=linprog(a['cost'],A_ub=a['matrix'],b_ub=a['rhs'],A_eq=a['equality'],b_eq=a['equality_rhs'],bounds=a['bounds'],method=SPEC['lp_method'],options=dict(time_limit=SPEC['lp_time_limit_s'],primal_feasibility_tolerance=SPEC['lp_primal_dual_tolerance'],dual_feasibility_tolerance=SPEC['lp_primal_dual_tolerance']))
  meta=dict(status=int(res.status),message=str(res.message),candidate=res.x is not None,success=bool(res.success),iterations=int(res.nit))
  if res.x is not None:
   x=res.x;lo=a['bounds'][:,0];hi=a['bounds'][:,1];violation=max(0.,float((a['matrix']@x-a['rhs']).max()),float(abs(a['equality']@x-a['equality_rhs']).max()),float((lo-x).max()),float((x-hi).max()));meta.update(independent_linear_violation=violation,linear_pass=violation<=SPEC['independent_linear_bound'],objective=float(a['cost']@x));np.save(folder/'witness.npy',x)
 meta.update(elapsed_s=time.monotonic()-start,mode=a['mode'],merit_ceiling=a['merit_ceiling']);(folder/'solver.json').write_text(json.dumps(meta,indent=2));return res,meta
