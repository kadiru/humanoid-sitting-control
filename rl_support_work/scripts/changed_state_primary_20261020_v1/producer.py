"""One cold solve of the newly linearized full-coordinate primary zero face."""
from common import *

def kkt(M,a,E,w,upper,lower):
 import numpy as np
 vals=M@w-a['rhs'];eq=E@w-a['equality_rhs']
 station=a['cost']+upper-lower
 lo=np.isfinite(a['bounds'][:,0]);hi=np.isfinite(a['bounds'][:,1])
 pr=max(0.,float(vals.max(initial=0)),float(abs(eq).max(initial=0)),
   float((a['bounds'][:,0]-w).max()),float((w-a['bounds'][:,1]).max()))
 du=float(abs(station).max());gap=abs(float(a['cost']@w+a['bounds'][hi,1]@upper[hi]-a['bounds'][lo,0]@lower[lo]))
 finite=all(np.isfinite(x).all() for x in [w,vals,eq,station,upper,lower,np.asarray([pr,du,gap])])
 signs=bool(np.all(lower>=0) and np.all(upper>=0) and np.all(lower[~lo]==0) and np.all(upper[~hi]==0))
 return dict(primal=pr,dual=du,absolute_gap=gap,finite=bool(finite),dual_signs=signs,
   passed=bool(finite and signs and max(pr,du,gap)<=1e-9)),dict(inequality=vals,equality=eq,stationarity=station)

def main():
 origin=admission('producer');spec=contract()
 for p,h in {**spec['input_sha256'],**spec['runtime_sha256']}.items():
  assert sha(p)==h,p
  check(origin)
 for folder in ['lifted_feasibility_audit_491','torque_path_audit_525']:
  assert read(D/folder/'independent_audit.json')['passed']
 assert read(PRIOR/'audit.json')['passed'] and read(PRIOR/'result.json')['trial_merit_accepted']
 event('input_identity_verified',origin)
 import numpy as np
 from scipy import sparse
 from builder import load,intervals,assemble
 d,f,y,J,drops,b=load();s=intervals(d,f,drops,1.0)
 p=assemble(d,b,s,None)
 np.savez(O/'state.npz',features=f,actualY=y,J=J,**{k:b[k] for k in ['r','e','h','guard']})
 for name,value in [('A',b['A']),('C',b['C']),('H',s['H'])]:sparse.save_npz(O/(name+'.npz'),value)
 write(O/'intervals.json',{**{key:[[x.numerator,x.denominator] for x in s[key]] for key in ['lower','upper','widths','closed_lower','closed_upper']},
   'owners':s['owners'],'drops':[[x.numerator,x.denominator] for x in drops],'trust':1.0})
 sparse.save_npz(O/'full_matrix.npz',p['matrix']);sparse.save_npz(O/'full_equality.npz',p['equality'])
 a={k:p[k] for k in ['rhs','cost','bounds','equality_rhs']};np.savez(O/'full_arrays.npz',**a)
 k=331495;m=68250;n=78790;keep=np.r_[np.arange(k),np.arange(k+2*m,p['matrix'].shape[0])]
 B=p['matrix'][keep].tocsr()
 E=sparse.vstack([p['equality'],sparse.hstack([b['C'],sparse.csr_matrix((m,1))]),
   sparse.csr_matrix(([1.],([0],[n])),shape=(1,n+1))],format='csr')
 na=dict(a,rhs=a['rhs'][keep],equality_rhs=np.r_[a['equality_rhs'],-b['h'],0.])
 sparse.save_npz(O/'native_matrix.npz',B);sparse.save_npz(O/'native_equality.npz',E);np.savez(O/'native_arrays.npz',**na)
 event('assembly_completed',origin)
 # Separate implementation, before any scipy solver invocation.
 from independent_input import prove
 prove(origin);event('independent_input_proof_completed',origin)
 from scipy.optimize import linprog
 remaining=CAP-(time.monotonic()-origin);assert remaining>0
 options=dict(time_limit=remaining,maxiter=4096,presolve=True,
   primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,ipm_optimality_tolerance=1e-9)
 write(O/'scipy_started.json',dict(method='highs-ipm',options=options,cold=True,
   x0_or_basis=False,input_proof_sha256=sha(O/'input_proof.json'),elapsed_s=time.monotonic()-origin))
 check(origin);start=time.monotonic()
 result=linprog(na['cost'],A_ub=B,b_ub=na['rhs'],A_eq=E,b_eq=na['equality_rhs'],bounds=na['bounds'],method='highs-ipm',options=options)
 elapsed=time.monotonic()-start
 write(O/'backend_result.json',dict(status=int(result.status),success=bool(result.success),
   message=str(result.message),candidate_returned=result.x is not None,iterations=int(result.nit),
   crossover_iterations=int(getattr(result,'crossover_nit',0)),seconds=elapsed,total_elapsed_s=time.monotonic()-origin))
 check(origin)
 if not(result.status==0 and result.success and result.x is not None):
  write(O/'candidate.json',dict(accepted=False,reason='No successful complete backend point',no_retry=True));return
 w=np.asarray(result.x);np.save(O/'backend_primal.npy',w)
 if not(w.shape==(78791,) and w.dtype==np.dtype('float64') and np.isfinite(w).all() and w[-1]==0):
  write(O/'candidate.json',dict(accepted=False,reason='Returned vector or unchanged zero scalar failed',no_retry=True));return
 dual=np.load(D/'zero_face_preparation_572/analytic_lower_bound_dual.npz');up=dual['upper'];lo=dual['lower']
 assert np.all(up==0) and np.all(lo[:-1]==0) and lo[-1]==1
 checks={}
 for name,M,aa,EE in [('full',p['matrix'],a,p['equality']),('native',B,na,E)]:
  checks[name],res=kkt(M,aa,EE,w,up,lo);np.savez(O/(name+'_residuals.npz'),**res)
  check(origin)
 passed=all(x['passed'] for x in checks.values())
 write(O/'original_checks.json',dict(passed=passed,systems=checks,all78791_backend_values_unchanged=True,
   analytic_bound_dual_inherited=True,zero_coordinate_lift=True,numerical_not_exact_optimality=True))
 if passed:
  predicted_rows=b['r']+b['A']@w[:-1];predicted_closure=b['h']+b['C']@w[:-1]
  soft=d['rows']['keep']&d['rows']['soft']
  theta=max(0.,float(-predicted_rows[soft].min()),float(abs(predicted_closure).max()))
  np.savez(O/'primary_prediction.npz',rows=predicted_rows,consistency=predicted_closure)
  write(O/'ceiling.json',dict(mu=b['phi'],theta=theta,ceiling=(b['phi']+theta)/2,eligible=theta<b['phi']))
  np.save(O/'witness.npy',w)
 write(O/'candidate.json',dict(accepted=passed,reason='Full and native numerical gates' if passed else 'Original gate failed',no_retry=True))
 event('producer_completed',origin)

if __name__=='__main__':main()
