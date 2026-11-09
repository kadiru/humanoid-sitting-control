"""New capped-maximum PDHG LP subsolver; no robot path launcher. Frozen539 inputs are read-only."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib,time,math
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
SPEC=dict(
 scope='Complete new LP SUBSOLVER for538 capped-maximum saddle form, with all native coordinates including auxiliary torques. Preparation of exact metric certificates and two NEW small LP tests only. No full-size LP or trajectory wrapper launched.',
 reference='Chambolle and Pock2010/2011, https://optimization-online.org/2010/06/2646/ . Exact-real reference is their fixed-step primal-dual iteration with theta1 after fixed positive diagonal metric coordinate substitution. No implementation or empirical performance is inherited.',
 input='H*x<=bH and g(S*x-bS), g(r)=max(L,maxr) on maxr<=U, infinity otherwise. Source538/539 copied-original rows and517 proof inherited by hash. Native costs must be zero, native bounds unbounded, equalities literally zero/RHS zero; reject unsupported nonzero domain, never silently drop it. No522 projection,504 transform,old rejected/internal vectors or source coefficient scaling.',
 metric='For K=vertical_stack(H,S), compute EXACT positive absolute row/column sums using integer units2^-1074. For each positive sum choose least enclosing power2^p, then sigma_i=2^(-p_i-1),tau_j=2^(-p_j-1); zero sum uses1. Reject nonfinite/zero step. Fixed once per NEW input; no adaptive balance/tuning. All source coefficients stay unchanged.',
 norm_proof='Exact sigma_i*rowL1_i<=1/2 and tau_j*colL1_j<=1/2. Cauchy gives ||sqrt(Sigma)*K*sqrt(T)*z||^2 <= sum_i sigma_i*rowL1_i*sum_j|Kij|*tau_j*z_j^2 <=1/4*||z||^2. Thus norm<=1/2<1. This proves a valid exact-real metric condition, NOT a finite-arithmetic convergence rate or45s success.',
 initialization='x=xbar=0,yH=lambda=0. Native increment zero is an LP seed, not physical hard feasibility. No warm start. Fixed theta1,no restart,acceleration,line search,ergodic averaging or best-iterate selection.',
 iteration='wH=yH+sigmaH*(H*xbar); yHnew=max(wH-sigmaH*bH,0). wS=lambda+sigmaS*(S*xbar); lambdanew=weighted prox of bS.lambda+g* at wS. xnew=x-T*(H.T*yHnew+S.T*lambdanew); xbarnew=2*xnew-x. All native coordinates remain joint. Cone projections concern dual variables only, no primal clipping/normalization.',
 finite_prox='q=w-sigma*(b+L), width=U-L. Select a=0, width, or sorted weighted-simplex threshold from q/sigma and longdouble prefix sums, then lambda=max(q-sigma*a,0). No bisection/retry. Check finite/nonnegative output, threshold bounds, scaled orthant stationarity and hinge complementarity<=1e-12. These numerical algebra screens are NOT an exact rational proximal certificate nor a summable-error convergence theorem. Independent small audit computes exact proximal primal-dual gaps of SAVED evaluations.',
 finite_limits='Exact-real convergence assumes an existing saddle point and exact arithmetic/proximals. Fixed finite screens and IEEE arithmetic do not establish that hypothesis or global convergence. Only explicit original unscaled numerical KKT may return a candidate. No empirical speed/conditioning/global infeasibility claim.',
 scalar='At every numerical screen v=max(L,max(S*x-bS)), without clipping to U; t=e*v. If KKT screens pass, compute every coupled residual with exactFraction source coefficients and binary64 x, take exact v=max(L,maxr), round ONCE nearest binary64 and t=e*v. Record exact maximum/owning row/rounding error and lower slack. Native x is BITWISE unchanged. A rounded scalar can lie below exact max; do NOT call it exact feasible. Recheck all source numerical KKT/bounds at unchanged thresholds; no nextafter or repair.',
 dual='s=sum(lambda); alpha=max(s-1,0),beta=max(1-s,0). Swap bound multipliers if e=-1. Source row duals are yH/lambda; numerical native stationarity and original scalar stationarity, source inequalities/bounds and absolute primal-dual gap all<=1e-9. Nonnegative finite duals required. Also source full inequalities<=1e-7. These are numerical KKT, not exact optimality.',
 stopping='First nonfinite,proximal check failure,unsupported domain,cap or deadline ends with NO candidate. No internal vector selected on failure. No infeasibility/unboundedness certificate. After one eligible scalar reconstruction, any final KKT failure terminates rather than polishing.',
 budget=dict(seconds_total=45.,maximum_PDHG_steps=4096,maximum_LP_calls_per_invocation=1,address_space_GiB=40,parent_wall_seconds=120),
 budget_scope='45s includes input checks/metric computation when not inherited/all PDHG/proximal and scalar certification. Deadline checked between operations and in exact residual loop; actual overshoot reported.40GiB/120s required for future full-size child. Current preparation only two declared small tests, same45s/4096caps once each.',
 trace='Bounded scalar ledger and ONE last transition by default. Full transition trace allowed ONLY for explicit small-test mode with n<=16,m<=64. No unbounded robot vector history.',
 future_admission='A returned source-native LP vector still needs FULLoriginal490 inequalities/equalities/bounds1e-7, actual-primary-derived interior ceiling, original unscaled eta>0 exact interval certification,immutable490 encoding,and actual model/J/unit/geometry/filter/runtime/PPO gates. Same full direction supplies phi/prediction; primary never mapped. No whole-path wrapper/independent actual-state auditor exists yet.',
 tests='Two NEW small LPs distinct from530/538 fixtures: asymmetric lower-bound primary with sum epigraph, and translated asymmetric clearance maximization. Exact primal/dual optima provided for independent certification. No old small LP/proximal example repeated.',physical_admission=False)
class Terminal(Exception):pass
def finite(*xs):
 if not all(np.all(np.isfinite(x)) for x in xs):raise Terminal('nonfinite arithmetic')
def exact_metric(H,S):
 K=sparse.vstack([H,S],format='csr');cols=[0]*K.shape[1];rp=[]
 for i in range(K.shape[0]):
  total=0
  for j,a in zip(K.indices[K.indptr[i]:K.indptr[i+1]],K.data[K.indptr[i]:K.indptr[i+1]]):
   num,den=abs(float(a)).as_integer_ratio();v=num<<(1074-(den.bit_length()-1));total+=v;cols[int(j)]+=v
  rp.append(-(total-1).bit_length()+1073 if total else 0)
 cp=[-(v-1).bit_length()+1073 if v else 0 for v in cols]
 rp=np.array(rp,dtype=np.int32);cp=np.array(cp,dtype=np.int32);sigma=np.ldexp(np.ones(len(rp)),rp);tau=np.ldexp(np.ones(len(cp)),cp)
 finite(sigma,tau)
 if not(np.all(sigma>0) and np.all(tau>0)):raise Terminal('unrepresentable diagonal step')
 return dict(row_power=rp,column_power=cp,sigma=sigma,tau=tau)
def weighted_prox(w,b,sigma,L,U):
 q=w-sigma*(b+L);width=U-L;finite(q);m0=float(np.sum(np.maximum(q,0),dtype=np.longdouble))
 if m0<=1:a=0.;branch=0
 else:
  mw=float(np.sum(np.maximum(q-sigma*width,0),dtype=np.longdouble))
  if mw>=1:a=width;branch=2
  else:
   ratios=q.astype(np.longdouble)/sigma.astype(np.longdouble);order=np.argsort(-ratios,kind='stable');sq=np.cumsum(q[order],dtype=np.longdouble);ss=np.cumsum(sigma[order],dtype=np.longdouble);threshold=(sq-1)/ss
   upper=ratios[order];lower=np.r_[upper[1:],np.longdouble(-np.inf)];ids=np.flatnonzero((threshold<=upper)&(threshold>=lower))
   if not len(ids):raise Terminal('weighted proximal active set absent')
   a=float(threshold[ids[0]]);branch=1
 lam=np.maximum(q-sigma*a,0);finite(lam,a)
 if not(0<=a<=width):raise Terminal('weighted proximal threshold outside hinge')
 grad=lam-w+sigma*(b+L+a);res=np.where(lam>0,abs(grad),np.maximum(-grad,0));scale=np.maximum(1,abs(lam)+abs(w)+abs(sigma*(b+L+a)))
 orth=float(np.max(res/scale,initial=0));mass=float(np.sum(lam,dtype=np.longdouble));hinge=width*max(mass-1,0)-a*(mass-1);hr=abs(hinge)/max(1,width*(mass+1))
 if not np.isfinite(hr) or max(orth,hr)>1e-12:raise Terminal('finite weighted proximal KKT screen failed')
 return lam,dict(threshold=a,branch=branch,mass=mass,orthant_relative=orth,hinge_relative=hr)
def metrics(H,S,bH,bS,x,y,lam,L,U,e,v_override=None):
 hx=H@x;sx=S@x;v=max(L,float(np.max(sx-bS,initial=-np.inf))) if v_override is None else v_override;s=float(np.sum(lam,dtype=np.longdouble));alpha=max(s-1,0);beta=max(1-s,0);tu,tl=(alpha,beta) if e==1 else(beta,alpha);t=e*v;lo,hi=sorted([e*L,e*U]);du=H.T@y+S.T@lam
 gap=e*t+float(bH@y+bS@lam)+hi*tu-lo*tl
 m=dict(primal=max(0.,float(np.max(hx-bH,initial=0)),float(np.max(sx-v-bS,initial=0)),lo-t,t-hi),dual=max(float(np.max(abs(du),initial=0)),abs(e-e*s+tu-tl)),gap=abs(gap),minimum_dual=min(float(np.min(y,initial=0)),float(np.min(lam,initial=0)),tu,tl),objective=e*t,scalar=t,canonical_scalar=v,upper_multiplier=tu,lower_multiplier=tl)
 finite(list(m.values()));return m
def solve_lp(H,S,bH,bS,L,U,e,out,metric=None,small_trace=False):
 start=time.monotonic();deadline=start+45;out.mkdir(exist_ok=False);records=[];history=[];last={};candidate=None;scalar_record=None;attempts=0
 def check():
  if time.monotonic()>deadline:raise Terminal('45s total solver wall deadline')
 try:
  H=H.tocsr();S=S.tocsr();n=H.shape[1];assert S.shape[1]==n and L<U and e in(-1,1)
  finite(H.data,S.data,bH,bS,L,U)
  if small_trace:assert n<=16 and H.shape[0]+S.shape[0]<=64
  if metric is None:metric=exact_metric(H,S)
  check();sigH=metric['sigma'][:len(bH)];sigS=metric['sigma'][len(bH):];tau=metric['tau'];x=np.zeros(n);xb=x.copy();y=np.zeros(len(bH));lam=np.zeros(len(bS))
  for k in range(4096):
   check();oldx=x;oldxb=xb;oldy=y;oldlam=lam;wH=y+sigH*(H@xb);wS=lam+sigS*(S@xb);y=np.maximum(wH-sigH*bH,0);lam,p=weighted_prox(wS,bS,sigS,L,U);x=oldx-tau*(H.T@y+S.T@lam);xb=2*x-oldx;finite(x,xb,y,lam);check()
   m=metrics(H,S,bH,bS,x,y,lam,L,U,e);last=dict(k=np.array(k),old_x=oldx,old_xbar=oldxb,old_y=oldy,old_lambda=oldlam,wH=wH,wS=wS,x=x.copy(),xbar=xb.copy(),y=y.copy(),lam=lam.copy(),threshold=np.array(p['threshold']))
   records.append(dict(k=k,proximal=p,metrics=m));
   if small_trace:history.append(last)
   check()
   if m['minimum_dual']>=0 and max(m['primal'],m['dual'],m['gap'])<=1e-9:
    attempts+=1;vx=list(map(lambda a:F(float(a)),x));g=F(float(L));owner=-1
    for i in range(S.shape[0]):
     if i%128==0:check()
     r=sum((F(float(a))*vx[int(j)] for j,a in zip(S.indices[S.indptr[i]:S.indptr[i+1]],S.data[S.indptr[i]:S.indptr[i+1]])),F(0))-F(float(bS[i]))
     if r>g:g=r;owner=i
    v=float(g);finite(v);scalar_record=dict(exact_canonical_scalar=str(g),coupled_owner=owner,once_rounded_canonical_scalar=v,rounding_error=str(F(v)-g),exact_lower_slack=str(F(v)-g),exact_upper_slack=str(F(float(U))-F(v)),native_coordinates_unmodified=True,exact_real_feasible_scalar=bool(F(v)>=g and F(v)<=F(float(U))))
    m=metrics(H,S,bH,bS,x,y,lam,L,U,e,v_override=v);check()
    if m['minimum_dual']<0 or max(m['primal'],m['dual'],m['gap'])>1e-9:raise Terminal('once-rounded scalar source KKT failed')
    candidate=dict(x=np.r_[x,e*v],hard_dual=y,coupled_dual=lam,upper_multiplier=np.array(m['upper_multiplier']),lower_multiplier=np.array(m['lower_multiplier']));reason='original-source numerical KKT witness';break
  else:reason='frozen4096 PDHG step cap'
 except Terminal as exc:reason=str(exc)
 except (AssertionError,ValueError,OverflowError,FloatingPointError) as exc:reason='arithmetic/input failure:'+repr(exc)
 elapsed=time.monotonic()-start
 if metric is not None:np.savez(out/'metric.npz',**metric)
 if last:np.savez_compressed(out/'last_step.npz',**last)
 if small_trace and history:np.savez_compressed(out/'small_trace.npz',**{k:np.stack([v[k] for v in history]) for k in history[0]})
 if scalar_record:(out/'scalar_reconstruction.json').write_text(json.dumps(scalar_record,indent=2))
 if candidate is not None:np.savez(out/'candidate.npz',**candidate);np.save(out/'witness.npy',candidate['x'])
 (out/'iteration_metrics.json').write_text(json.dumps(records,indent=2))
 r=dict(status=0 if candidate is not None else 1,success=candidate is not None,candidate=candidate is not None,message=reason,elapsed_s=elapsed,elapsed_including_evidence_s=time.monotonic()-start,iterations=len(records),weighted_proximal_evaluations=len(records),scalar_reconstruction_attempts=attempts,actual_lp_call=True,full_size_robot_LP=False,small_test=small_trace,physical_admission=False,linear_system_solves=0,gram_factorizations=0,method='capped-maximum-native-PDHG',trace_policy='all SMALL transitions only when explicitly bounded; otherwise last transition plus scalar ledger',budget=SPEC['budget'])
 (out/'result.json').write_text(json.dumps(r,indent=2));return r
