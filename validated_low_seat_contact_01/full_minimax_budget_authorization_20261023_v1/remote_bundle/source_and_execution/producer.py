"""Proposed single full-original minimax solve; blocked pending scoped approval."""
from common import *

def useful_improvement(mu, theta):
 from fractions import Fraction as F
 import math
 return bool(math.isfinite(mu) and math.isfinite(theta) and mu>0 and theta>=0 and 100*(F(mu)-F(theta))>=F(mu))

def verify_input(origin):
 proof=read(PRIOR/'input_proof.json')
 assert proof['passed'] and proof['new_state_full_input_proved'] and not proof['old_state_reductions_used']
 assert read(ACCEPTED/'audit.json')['passed'] and read(ACCEPTED/'result.json')['trial_merit_accepted']
 for name in ['full_matrix.npz','full_arrays.npz','full_equality.npz','state.npz','A.npz','C.npz','H.npz','intervals.json']:
  assert sha(PRIOR/name)==proof['file_sha256'][name],name
  check(origin)
 import numpy as np
 from scipy import sparse
 M=sparse.load_npz(PRIOR/'full_matrix.npz').tocsr()
 E=sparse.load_npz(PRIOR/'full_equality.npz').tocsr();a=dict(np.load(PRIOR/'full_arrays.npz'))
 assert M.shape==(652529,78791) and M.nnz==2614306
 assert E.shape==(1366,78791) and E.nnz==0 and np.all(a['equality_rhs']==0)
 assert np.count_nonzero(a['cost'])==1 and a['cost'][-1]==1
 assert np.all(np.isneginf(a['bounds'][:-1,0])) and np.all(np.isposinf(a['bounds'][:-1,1]))
 assert np.array_equal(a['bounds'][-1],[0.,1.])
 assert np.isfinite(M.data).all() and np.isfinite(a['rhs']).all()
 write(O/'input_identity.json',dict(passed=True,inherited_full_input_proof_sha256=sha(PRIOR/'input_proof.json'),
   no_new_assembly_or_proof_replay=True,full_rows=652529,columns=78791,nnz=2614306,
   equality_rows=1366,equality_nnz=0,added_zero_face_absent=True,
   all_original_consistency_inequalities_present=True,physical_gates_unchanged=True,
   prior_terminal_zero_face_unchanged=True))
 event('full_input_identity_verified',origin)
 return M,E,a

def check_point(M,E,a,w,marginal):
 import numpy as np
 y=-marginal['inequality'];nu=-marginal['equality'];up=-marginal['upper'];low=marginal['lower']
 values=M@w-a['rhs'];eq=E@w-a['equality_rhs']
 station=a['cost']+M.T@y+E.T@nu+up-low
 lf=np.isfinite(a['bounds'][:,0]);uf=np.isfinite(a['bounds'][:,1])
 primal=max(0.,float(values.max(initial=0)),float(abs(eq).max(initial=0)),
   float((a['bounds'][:,0]-w).max()),float((w-a['bounds'][:,1]).max()))
 dual=float(abs(station).max(initial=0))
 gap=abs(float(a['cost']@w+a['rhs']@y+a['equality_rhs']@nu+a['bounds'][uf,1]@up[uf]-a['bounds'][lf,0]@low[lf]))
 finite=all(np.isfinite(v).all() for v in [w,y,nu,up,low,values,eq,station,np.asarray([primal,dual,gap])])
 signs=bool(np.all(y>=0) and np.all(up>=0) and np.all(low>=0) and np.all(up[~uf]==0) and np.all(low[~lf]==0))
 return dict(passed=bool(finite and signs and max(primal,dual,gap)<=1e-9),primal=primal,dual=dual,
   absolute_gap=gap,finite=bool(finite),dual_signs=signs),dict(inequality=values,equality=eq,stationarity=station)

def main():
 origin=admission('producer');spec=contract()
 for path,h in {**spec['input_sha256'],**spec['runtime_sha256']}.items():
  assert sha(path)==h,path
  check(origin)
 M,E,a=verify_input(origin)
 import numpy as np
 from scipy import sparse
 from scipy.optimize import linprog
 remaining=SOLVE_END-(time.monotonic()-origin);assert remaining>0
 options=dict(time_limit=remaining,maxiter=4096,presolve=True,
   primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,ipm_optimality_tolerance=1e-9)
 write(O/'scipy_started.json',dict(method='highs-ipm',options=options,cold=True,x0_or_basis=False,
   input_identity_sha256=sha(O/'input_identity.json'),elapsed_s=time.monotonic()-origin))
 check(origin);start=time.monotonic()
 result=linprog(a['cost'],A_ub=M,b_ub=a['rhs'],A_eq=E,b_eq=a['equality_rhs'],bounds=a['bounds'],method='highs-ipm',options=options)
 write(O/'backend_result.json',dict(status=int(result.status),success=bool(result.success),message=str(result.message),
   candidate_returned=result.x is not None,iterations=int(result.nit),crossover_iterations=int(getattr(result,'crossover_nit',0)),
   seconds=time.monotonic()-start,total_elapsed_s=time.monotonic()-origin))
 check(origin)
 if not(result.status==0 and result.success and result.x is not None):
  write(O/'candidate.json',dict(accepted=False,reason='Backend did not return a successful complete point',no_retry=True));return
 w=np.asarray(result.x);np.save(O/'backend_primal.npy',w)
 assert w.dtype==np.dtype('float64') and w.shape==(78791,) and np.isfinite(w).all()
 marginal=dict(inequality=np.asarray(result.ineqlin.marginals),equality=np.asarray(result.eqlin.marginals),
   upper=np.asarray(result.upper.marginals),lower=np.asarray(result.lower.marginals))
 np.savez(O/'backend_marginals.npz',**marginal)
 checks,res=check_point(M,E,a,w,marginal);write(O/'original_checks.json',checks)
 np.savez(O/'residuals.npz',**res);check(origin)
 if not checks['passed']:
  write(O/'candidate.json',dict(accepted=False,reason='Full-original numerical KKT gate failed',no_retry=True));return
 state=np.load(PRIOR/'state.npz');A=sparse.load_npz(PRIOR/'A.npz');C=sparse.load_npz(PRIOR/'C.npz')
 row=state['r']+A@w[:-1];h=state['h']+C@w[:-1]
 ra=np.load(D/'clock_row_preparation_v2_446/row_arrays.npz');soft=ra['keep']&ra['soft']
 theta=max(0.,float(-row[soft].min()),float(abs(h).max()))
 mu=read(ACCEPTED/'decision.json')['summary']['phi']
 assert np.isfinite(row).all() and np.isfinite(h).all() and np.isfinite(theta) and np.isfinite(mu)
 eligible=useful_improvement(mu,theta)
 np.savez(O/'primary_prediction.npz',rows=row,consistency=h)
 write(O/'ceiling.json',dict(mu=mu,theta=theta,ceiling=(mu+theta)/2 if eligible else None,
   eligible=eligible,epigraph=float(w[-1]),epigraph_is_not_physical_permission=True))
 if eligible:np.save(O/'witness.npy',w)
 write(O/'candidate.json',dict(accepted=eligible,full_numerical_checks_passed=True,
   reason='At least one percent actual modeled improvement' if eligible else 'Less than one percent modeled improvement',no_retry=True))
 event('producer_completed',origin)

if __name__=='__main__':main()
