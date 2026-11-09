"""Independent certificate and failure accounting; no producer/optimizer imports."""
from common import *
def independent_check(M,E,a,w,m):
 import numpy as np
 y=np.negative(m['inequality']);nu=np.negative(m['equality']);upper=np.negative(m['upper']);lower=m['lower']
 residual=M.dot(w)-a['rhs'];eq=E.dot(w)-a['equality_rhs']
 gradient=a['cost']+M.transpose().dot(y);gradient=gradient+E.transpose().dot(nu)+upper-lower
 lf=np.isfinite(a['bounds'][:,0]);uf=np.isfinite(a['bounds'][:,1])
 primal=max(0.,float(np.max(residual,initial=0)),float(np.max(abs(eq),initial=0)),float(np.max(a['bounds'][:,0]-w)),float(np.max(w-a['bounds'][:,1])))
 dual=float(np.max(abs(gradient),initial=0))
 gap=abs(float(a['cost'].dot(w)+a['rhs'].dot(y)+a['equality_rhs'].dot(nu)+a['bounds'][uf,1].dot(upper[uf])-a['bounds'][lf,0].dot(lower[lf])))
 finite=all(np.isfinite(x).all() for x in [w,y,nu,upper,lower,residual,eq,gradient,np.asarray([primal,dual,gap])])
 signs=bool(np.all(y>=0) and np.all(upper>=0) and np.all(lower>=0) and np.all(upper[~uf]==0) and np.all(lower[~lf]==0))
 checks=dict(passed=bool(finite and signs and max(primal,dual,gap)<=1e-9),primal=primal,dual=dual,absolute_gap=gap,finite=bool(finite),dual_signs=signs)
 return checks,dict(inequality=residual,equality=eq,stationarity=gradient)

def main():
 origin=admission('audit');spec=contract()
 for path,h in {**spec['input_sha256'],**spec['runtime_sha256']}.items():assert sha(path)==h,path;check(origin)
 pe=read(O/'producer_exit.json')
 if not pe['completed_within_deadline']:
  write(O/'audit.json',dict(passed=True,candidate_accepted=False,scope='Terminal producer failure/deadline accounting only; any partial witness remains unadmitted',producer_exit=pe,unadmitted_witness_file_present=(O/'witness.npy').exists(),physical_admission=False));return
 decision=read(O/'candidate.json');proof=read(O/'input_proof.json');tp=read(O/'target_input_proof.json')
 assert proof['passed'] and proof['new_state_full_input_proved'] and not proof['zero_face_added'] and tp['passed']
 for n,h in proof['file_sha256'].items():assert sha(O/n)==h,n
 for file,key in [('full_matrix.npz','full_matrix_sha256'),('target_matrix.npz','target_matrix_sha256'),('target_rhs.npy','target_rhs_sha256'),('augmented_matrix.npz','augmented_matrix_sha256'),('augmented_arrays.npz','augmented_arrays_sha256')]:assert sha(O/file)==tp[key]
 assert sha(PRIOR/'accepted_J.npz')==tp['source_J_sha256'] and sha(ATTR/'attribution.json')==tp['source_attribution_sha256']
 if not (O/'scipy_started.json').exists():
  assert not decision['accepted'] and not (O/'witness.npy').exists()
  write(O/'audit.json',dict(passed=True,candidate_accepted=False,scope='Assembly consumed solver allowance; no solve',physical_admission=False));return
 marker=read(O/'scipy_started.json');backend=read(O/'backend_result.json')
 assert marker['method']=='highs-ipm' and marker['cold'] and not marker['x0_or_basis']
 assert 0<marker['options']['time_limit']<=SOLVE_END and marker['options']['maxiter']==4096
 for k in ['primal_feasibility_tolerance','dual_feasibility_tolerance','ipm_optimality_tolerance']:assert marker['options'][k]==1e-9
 if not (O/'checks.json').exists():
  assert not(backend['success'] and backend['status']==0 and backend['candidate_returned'])
  assert not decision['accepted'] and not (O/'witness.npy').exists()
  write(O/'audit.json',dict(passed=True,candidate_accepted=False,scope='Terminal backend failure accounting; no point admitted',physical_admission=False));return
 import numpy as np
 from scipy import sparse
 from independent_input import csr,same
 M=sparse.load_npz(O/'full_matrix.npz');G=sparse.load_npz(O/'target_matrix.npz');T=sparse.load_npz(O/'augmented_matrix.npz');E=sparse.load_npz(O/'full_equality.npz')
 a=dict(np.load(O/'full_arrays.npz'));aa=dict(np.load(O/'augmented_arrays.npz'));grhs=np.load(O/'target_rhs.npy')
 csr(T[:652529],M);csr(T[652529:],G);same(aa['rhs'],np.r_[a['rhs'],grhs])
 w=np.load(O/'backend_primal.npy');m=dict(np.load(O/'backend_marginals.npz'))
 assert w.dtype==np.float64 and w.shape==(78791,)
 for name,n in [('inequality',652531),('equality',1366),('upper',78791),('lower',78791)]:assert m[name].shape==(n,) and m[name].dtype==np.float64
 aug,ar=independent_check(T,E,aa,w,m);original,orr=independent_check(M,E,a,w,dict(m,inequality=m['inequality'][:652529]))
 zero=all(float(x)==0.0 for x in m['inequality'][652529:])
 for tag,res in [('augmented',ar),('original',orr)]:
  saved=np.load(O/(tag+'_residuals.npz'))
  for k,v in res.items():same(v,saved[k])
 target=grhs-G.dot(w);same(target,np.load(O/'target_prediction.npy'));target_pass=all(float(x)>=0 for x in target)
 state=np.load(O/'state.npz');A=sparse.load_npz(O/'A.npz');C=sparse.load_npz(O/'C.npz');row=state['r']+A.dot(w[:-1]);h=state['h']+C.dot(w[:-1]);pred=np.load(O/'primary_prediction.npz');same(row,pred['rows']);same(h,pred['consistency'])
 ra=np.load(D/'clock_row_preparation_v2_446/row_arrays.npz');soft=ra['keep']&ra['soft'];theta=max(0.,float(-row[soft].min()),float(abs(h).max()));mu=float(read(PRIOR/'decision.json')['summary']['phi'])
 from fractions import Fraction
 useful=bool(np.isfinite(theta) and theta>=0 and Fraction(theta)<=Fraction(mu)*Fraction(99,100))
 checks=dict(augmented=aug,original=original,added_marginals_exact_zero=zero,target_prediction_pass=target_pass,one_percent_improvement=useful,mu=mu,theta=theta)
 assert checks==read(O/'checks.json')
 accepted=bool(aug['passed'] and original['passed'] and zero and target_pass and useful)
 assert decision['accepted']==accepted
 if accepted:same(w,np.load(O/'witness.npy'))
 else:assert not (O/'witness.npy').exists()
 write(O/'audit.json',dict(passed=True,candidate_accepted=accepted,full_original_checks=original,augmented_checks=aug,added_marginals_exact_zero=zero,unchanged_original_dual_subvector=True,scope='Independent full-original and augmented numerical certificate; not exact optimality or physical admission',physical_admission=False));check(origin)
if __name__=='__main__':main()
