"""New bounded HSD splitting LP subsolver using the inherited528 Schur structure.

No complete robot-path launcher. Preparation runs only two NEW small LP tests.
"""
from pathlib import Path
import json,hashlib,time
import numpy as np
from scipy import sparse,linalg
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
SPEC=dict(
 scope='NEW complete bounded LP SUBSOLVER: homogeneous self-dual Douglas-Rachford splitting with state-local Schur/PCG resolvents. Preparation and two new small LP tests only; no full-size LP or robot trajectory launcher.',
 reference='Algorithmic starting point: O Donoghue,Chu,Parikh,Boyd2016 https://web.stanford.edu/~boyd/papers/pdf/scs.pdf sections2-4. Official linear-system context https://www.cvxgrp.org/scs/linear_solver/index.html . This custom implementation is not SCS or HiGHS, and no performance/convergence claim is inherited from those implementations.',
 cone='Append EVERY finite source variable upper/lower bound as an unchanged signed inequality. Source equality must have literally zero stored coefficients and zero RHS; otherwise reject unsupported input, never drop nonzero equalities. Their source provenance and final original checks remain. For A*x+s=b,s>=0,dual A.T*y+c=0,y>=0.',
 scaling='One deterministic positive dyadic row scaling based on max(1,rowL1,abs(b)), then dyadic column scaling based on columnL2 norm. No iterative balancing, tuning or coefficient threshold. Preserve every nonzero exactly under each power-of-two multiplication; reject nonfinite/underflow/inexact scale. x_original=column_scale*x_scaled, y_original=row_scale*y_scaled. All final tolerances use ORIGINAL unscaled cone rows/cost, not just scaled residuals.',
 embedding='u=(x,y,tau),v=(r,s,kappa);Q=[[0,A.T,c],[-A,0,b],[-c.T,-b.T,0]]. C=R^n x nonnegative^m x nonnegative. Q is skew. For tau>0,exact v=Q*u,u in C,v in Cdual implies primal/dual feasibility and zero gap after division by tau. Finite residuals are checked explicitly; zero/tiny tau never certifies a witness.',
 initialization='All x,y,r,s zero; u_tau=v_kappa=1. No old primary,partial,rejected/internal vector or saved528 example input. Fixed alpha1, identity DR metric; no adaptive relaxation,barrier parameters,regularization or restart.',
 iteration='tilde=(I+Q)^-1(u+v);newu=projection_C(tilde-v);newv=v-tilde+newu. The projection only sets solver cone coordinates y,tau nonnegative; it NEVER clips returned retained primal coordinates, physical loads,torques,gaps or commands. No iterate normalization or post-hoc witness repair.',
 resolvent='Let M=[[I,A.T],[-A,I]],q=(c,b). Solve M*p=(rx,ry) using K=I+A.T*A: px=K^-1(rx-A.T*ry),py=ry+A*px. Cache z=M^-1*q once. tau=(rtau+q.T*p)/(1+q.T*z),xy=p-z*tau. Check the FULL original (I+Q)*tilde-r residual, including cached z error, before every DR step. No unchecked rank-one shortcut.',
 factorization='Use inherited528 local/separator columns and17-block support after positive column/row scaling. Appended finite bounds on epigraph/eta are separator-only. New local Gram blocks I+B.T*B are factorized once for each NEW solver input, not old mechanics L/K/Q. No changing barrier weights: this fixed resolvent identity needs no regularization-to-zero schedule. S uses sparse B,C products plus local Cholesky solves, never forms K,V,T orS.',
 separator='Zero-start preconditioned CG solves each separator equation. Fixed positive preconditioner diag(T)=1+column_sum(C^2), not a claimed Schur diagonal. Check positive finite curvature and stop on nonpositive/nonfinite arithmetic. Check FULL recovered K equation residual; no inexact-solve refinement/retry. Every inner product/matvec and local solve is accounted.',
 inexactness='DR step k requires Euclidean FULL resolvent residual <=2^-20/(k+1)^2; this sequence is summable. Cached q solve is checked at1e-12. Inner K solve targets full residual <=eps/8 and CG recurrence eps/64; independently recomputed K/resolvent residuals govern. Floating failure stops, not a claim of global convergence. No infinite-iteration or accuracy guarantee under finite resources.',
 termination='Only return a candidate when ORIGINAL unscaled cone primal equality residual,dual equality residual and absolute duality gap are ALL<=1e-9, with finite x,y,s and y,s>=0. Also test original inequalities/bounds<=1e-7. These are numerical KKT checks, not exact rational optimality. No infeasibility/unboundedness/global impossibility claim is produced. First nonfinite,linear-solve failure,iteration/time cap terminates without candidate; internal vectors saved only as rejected diagnostics.',
 budget=dict(seconds_total=45.,maximum_DR_iterations=4096,maximum_CG_iterations_per_K_solve=256,maximum_total_CG_iterations=16384,maximum_LP_calls_per_invocation=1,parent_wall_seconds=120,address_space_GiB=40),
 budget_scope='45s includes scaling,local factors,cached q solve and all DR/CG work; no separate extra linear-solver budget. Parent120s/40GiB are required for a future full-size child. Preparation invokes only two declared small tests with the same45s/caps, once each. No main full-size solve or new trajectory loop launched.',
 admission='A numerical KKT return is only a projected LP result. Unmodified retained coordinates still require522 exact endpoint/midpoint binary64 torque backsubstitution,source and FULL ORIGINAL490<=1e-7,actual-primary-derived witness ceiling,exact original interval eta>0 and immutable490 finite encoding. Actual mechanics/geometry/filter/runtime/physics/PPO criteria unchanged. Complete full-path wrapper and independent actual-step auditor must be frozen separately.',
 tests='Two NEW small analytically bounded LPs with explicit exact primal/dual certificates. They test infeasible-start primary-like minimization and clearance-like maximization; NOT actual robot LPs,old522 interval examples,old516 fixture,actual candidate vectors or newphysical evidence.',
 physical_admission=False)

class Terminal(Exception):pass

def exact_scale(data,powers):
 out=np.ldexp(data,powers);nz=data!=0
 assert np.all(np.isfinite(out)) and np.all(out[nz]!=0)
 m,e=np.frexp(data[nz]);mm,ee=np.frexp(out[nz]);assert np.array_equal(m,mm) and np.array_equal(ee-e,np.broadcast_to(powers,data.shape)[nz])
 return out

def normalize(A,b,c,bounds,E,e):
 A=A.tocsr();assert np.all(E.data==0) and np.all(e==0), 'Nonzero equality unsupported; never silently omitted'
 rows=[];cols=[];values=[];br=[];owners=[]
 for j,(lo,hi) in enumerate(bounds):
  if np.isfinite(hi):rows.append(len(br));cols.append(j);values.append(1.);br.append(hi);owners.append((j,1))
  if np.isfinite(lo):rows.append(len(br));cols.append(j);values.append(-1.);br.append(-lo);owners.append((j,-1))
 A0=sparse.vstack([A,sparse.csr_matrix((values,(rows,cols)),shape=(len(br),A.shape[1]))],format='csr');b0=np.r_[b,br]
 row_power=-np.frexp(np.maximum.reduce([np.ones(len(b0)),np.asarray(abs(A0).sum(axis=1)).ravel(),abs(b0)]))[1]
 rid=np.repeat(np.arange(A0.shape[0]),np.diff(A0.indptr));Ar=A0.copy();Ar.data=exact_scale(A0.data,row_power[rid]);bs=exact_scale(b0,row_power)
 cn=np.sqrt(np.asarray(Ar.power(2).sum(axis=0)).ravel());column_power=np.where(cn>0,-np.frexp(cn)[1],0)
 At=Ar.copy();At.data=exact_scale(Ar.data,column_power[Ar.indices]);cs=exact_scale(c,column_power)
 return dict(original=A0,original_rhs=b0,original_cost=c,matrix=At,rhs=bs,cost=cs,row_power=row_power,column_power=column_power,bound_owners=np.asarray(owners,dtype=np.int64).reshape(-1,2))

class Schur:
 def __init__(self,A,local,separator,ptr,deadline,stats):
  self.A=A;self.local=local;self.separator=separator;self.ptr=ptr;self.B=A[:,local].tocsr();self.C=A[:,separator].tocsr();self.deadline=deadline;self.stats=stats;self.factors=[]
  self.preconditioner=1+np.asarray(self.C.power(2).sum(axis=0)).ravel();bc=self.B.tocsc()
  for lo,hi in zip(ptr[:-1],ptr[1:]):
   self.check();rows=np.unique(bc.indices[bc.indptr[lo]:bc.indptr[hi]]);b=self.B[rows,lo:hi]
   self.factors.append(linalg.cholesky(np.eye(hi-lo)+(b.T@b).toarray(),lower=True,check_finite=True));stats['local_cholesky_blocks']+=1
 def check(self):
  if time.monotonic()>self.deadline:raise Terminal('45s total solver wall deadline')
 def local_solve(self,b):
  self.check();x=np.empty_like(b)
  for k,(lo,hi) in enumerate(zip(self.ptr[:-1],self.ptr[1:])):x[lo:hi]=linalg.cho_solve((self.factors[k],True),b[lo:hi],check_finite=True)
  self.stats['local_rhs_blocks']+=len(self.factors);return x
 def apply(self,y):
  self.check();self.stats['schur_matvecs']+=1;cy=self.C@y
  return y+self.C.T@(cy-self.B@self.local_solve(self.B.T@cy))
 def solve(self,rhs,eps):
  self.check();self.stats['K_solves']+=1;rl=rhs[self.local];rs=rhs[self.separator];q=self.local_solve(rl);d=rs-self.C.T@(self.B@q)
  x=np.zeros(len(rs));r=d.copy();z=r/self.preconditioner;p=z.copy();rz=float(r@z);count=0
  while np.linalg.norm(r)>eps/8:
   if count>=256 or self.stats['CG_iterations']>=16384:raise Terminal('frozen CG iteration cap')
   ap=self.apply(p);den=float(p@ap)
   if not np.isfinite(den) or den<=0 or not np.isfinite(rz):raise Terminal('nonpositive/nonfinite CG curvature')
   alpha=rz/den;x+=alpha*p;r-=alpha*ap;z=r/self.preconditioner;new=float(r@z);p=z+(new/rz)*p;rz=new;count+=1;self.stats['CG_iterations']+=1
  ans=np.empty_like(rhs);ans[self.separator]=x;ans[self.local]=self.local_solve(rl-self.B.T@(self.C@x))
  residual=float(np.linalg.norm(ans+self.A.T@(self.A@ans)-rhs));self.stats['maximum_K_residual']=max(residual,self.stats['maximum_K_residual'])
  if not np.isfinite(residual) or residual>eps:raise Terminal('full recovered K residual failed')
  return ans,residual,count

def solve_lp(A,b,c,bounds,E,e,local,separator,ptr,out,problem_scope='small_arithmetic_fixture'):
 start=time.monotonic();deadline=start+45;stats=dict(local_cholesky_blocks=0,local_rhs_blocks=0,K_solves=0,CG_iterations=0,schur_matvecs=0,maximum_K_residual=0.,DR_iterations=0);records=[];candidate=None;norm=None;solver=None
 try:
  norm=normalize(A,b,c,bounds,E,e);At=norm['matrix'];bt=norm['rhs'];ct=norm['cost'];n=A.shape[1];m=At.shape[0]
  solver=Schur(At,local,separator,ptr,deadline,stats)
  def minv(rx,ry,eps):
   px,err,cg=solver.solve(rx-At.T@ry,eps);return px,ry+At@px,err,cg
  zx,zy,_,_=minv(ct,bt,1e-13);zr=np.r_[zx+At.T@zy-ct,zy-At@zx-bt]
  if np.linalg.norm(zr)>1e-12:raise Terminal('cached M inverse residual failed')
  den=1+ct@zx+bt@zy
  if not np.isfinite(den) or den<=0:raise Terminal('cached positive denominator failed')
  u=np.zeros(n+m+1);v=u.copy();u[-1]=v[-1]=1.
  for k in range(4096):
   solver.check();eps=2.**-20/(k+1)**2;r=u+v;px,py,ke,cgi=minv(r[:n],r[n:n+m],eps/8)
   tau=(r[-1]+ct@px+bt@py)/den;tilde=np.r_[px-zx*tau,py-zy*tau,tau]
   residual=np.r_[tilde[:n]+At.T@tilde[n:n+m]+ct*tilde[-1]-r[:n],tilde[n:n+m]-At@tilde[:n]+bt*tilde[-1]-r[n:n+m],tilde[-1]-ct@tilde[:n]-bt@tilde[n:n+m]-r[-1]]
   err=float(np.linalg.norm(residual))
   if not np.isfinite(err) or err>eps:raise Terminal('full HSD resolvent residual failed')
   old_u=u.copy();old_v=v.copy();u=tilde-v;u[n:]=np.maximum(u[n:],0.);v=v-tilde+u;stats['DR_iterations']=k+1
   metrics=None
   if u[-1]>0:
    x=np.ldexp(u[:n]/u[-1],norm['column_power']);y=np.ldexp(u[n:n+m]/u[-1],norm['row_power']);s=np.ldexp(v[n:n+m]/u[-1],-norm['row_power'])
    A0=norm['original'];b0=norm['original_rhs'];pr=A0@x+s-b0;du=A0.T@y+c;gap=float(c@x+b0@y)
    metrics=dict(primal=float(np.max(abs(pr),initial=0)),dual=float(np.max(abs(du),initial=0)),gap=abs(gap),inequality=float(max(0,np.max(A0@x-b0,initial=0))),min_dual=float(np.min(y,initial=0)),min_slack=float(np.min(s,initial=0)))
    finite=all(np.all(np.isfinite(a)) for a in [x,y,s]);passed=finite and min(metrics['min_dual'],metrics['min_slack'])>=0 and max(metrics['primal'],metrics['dual'],metrics['gap'])<=1e-9 and metrics['inequality']<=1e-7
    if passed:candidate=dict(x=x,y=y,slack=s)
   records.append(dict(k=k,epsilon=eps,resolvent_residual=err,K_residual=ke,CG_iterations=cgi,metrics=metrics,old_u=old_u,old_v=old_v,tilde=tilde.copy(),u=u.copy(),v=v.copy()))
   if candidate is not None:break
  reason='numerical primal-dual witness' if candidate is not None else 'frozen DR iteration cap'
 except Terminal as exc:reason=str(exc)
 except (np.linalg.LinAlgError,ValueError,FloatingPointError) as exc:reason='arithmetic failure:'+repr(exc)
 if norm is not None:
  sparse.save_npz(out/'scaled_matrix.npz',norm['matrix']);sparse.save_npz(out/'cone_original.npz',norm['original']);np.savez_compressed(out/'scaled_arrays.npz',rhs=norm['rhs'],cost=norm['cost'],original_rhs=norm['original_rhs'],original_cost=norm['original_cost'],row_power=norm['row_power'],column_power=norm['column_power'],bound_owners=norm['bound_owners'])
 if records:
  np.savez_compressed(out/'iterations.npz',**{key:np.asarray([r[key] for r in records]) for key in ['old_u','old_v','tilde','u','v']})
  (out/'iteration_metrics.json').write_text(json.dumps([{key:value for key,value in r.items() if key not in ['old_u','old_v','tilde','u','v']} for r in records],indent=2))
 if candidate is not None:np.savez_compressed(out/'candidate.npz',**candidate)
 result=dict(status=0 if candidate is not None else 1,reason=reason,elapsed=time.monotonic()-start,statistics=stats,candidate_returned=candidate is not None,problem_scope=problem_scope,full_size_robot_LP=problem_scope=='robot_projected_LP',physical_admission=False)
 (out/'result.json').write_text(json.dumps(result,indent=2));return result

def fixtures():
 return [dict(name='infeasible_start_primary',A=np.array([[-1.,0,0],[0,-1,0],[1,0,-1],[0,1,-1],[0,0,1]]),b=np.array([-1.,-2,0,0,3]),c=np.array([0.,0,1]),bounds=np.array([[-np.inf,np.inf],[-np.inf,np.inf],[-4,4]]),exact_x=np.array([1.,2,2]),exact_y=np.array([0.,1,0,1,0,0,0])),dict(name='clearance_maximization',A=np.array([[1.,0,1],[-1,0,1],[0,2,1],[0,-2,1]]),b=np.array([1.,1,2,2]),c=np.array([0.,0,-1]),bounds=np.array([[-np.inf,np.inf],[-np.inf,np.inf],[0,1.5]]),exact_x=np.array([0.,0,1]),exact_y=np.array([.5,.5,0,0,0,0]))]
