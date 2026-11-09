"""Independent full-space KKT/prediction audit, without producer or optimizer imports."""
from common import *

def main():
 origin=admission('audit');spec=contract()
 for path,h in {**spec['input_sha256'],**spec['runtime_sha256']}.items():
  assert sha(path)==h,path
  check(origin)
 assert read(O/'producer_exit.json')['completed_within_deadline']
 identity=read(O/'input_identity.json');proof=read(PRIOR/'input_proof.json')
 assert identity['passed'] and proof['passed'] and proof['new_state_full_input_proved']
 assert identity['inherited_full_input_proof_sha256']==sha(PRIOR/'input_proof.json')
 for name in ['full_matrix.npz','full_arrays.npz','full_equality.npz','state.npz','A.npz','C.npz']:
  assert sha(PRIOR/name)==proof['file_sha256'][name]
 marker=read(O/'scipy_started.json');backend=read(O/'backend_result.json');decision=read(O/'candidate.json')
 assert marker['method']=='highs-ipm' and marker['cold'] and not marker['x0_or_basis']
 assert marker['input_identity_sha256']==sha(O/'input_identity.json')
 if not (O/'original_checks.json').exists():
  assert not (backend['status']==0 and backend['success'] and backend['candidate_returned'])
  assert not decision['accepted'] and not (O/'witness.npy').exists()
  write(O/'audit.json',dict(passed=True,candidate_accepted=False,scope='Failure accounting only; no accepted point',physical_admission=False));return
 import numpy as np
 from scipy import sparse
 a=dict(np.load(PRIOR/'full_arrays.npz'));M=sparse.load_npz(PRIOR/'full_matrix.npz');E=sparse.load_npz(PRIOR/'full_equality.npz')
 assert backend['status']==0 and backend['success'] and backend['candidate_returned']
 assert M.shape==(652529,78791) and M.nnz==2614306 and E.shape==(1366,78791) and E.nnz==0
 w=np.load(O/'backend_primal.npy');m=np.load(O/'backend_marginals.npz')
 assert w.dtype==np.dtype('float64') and w.shape==(78791,)
 for name,n in [('inequality',652529),('equality',1366),('upper',78791),('lower',78791)]:
  assert m[name].shape==(n,) and m[name].dtype==np.dtype('float64')
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
 assert checks==read(O/'original_checks.json')
 r=np.load(O/'residuals.npz')
 for k,v in [('inequality',residual),('equality',eq),('stationarity',gradient)]:assert np.array_equal(r[k],v)
 if not checks['passed']:
  assert not decision['accepted'] and not (O/'witness.npy').exists()
  write(O/'audit.json',dict(passed=True,candidate_accepted=False,full_checks=checks,scope='Rejected point audited',physical_admission=False));return
 state=np.load(PRIOR/'state.npz');A=sparse.load_npz(PRIOR/'A.npz');C=sparse.load_npz(PRIOR/'C.npz')
 row=state['r']+A.dot(w[:-1]);h=state['h']+C.dot(w[:-1]);saved=np.load(O/'primary_prediction.npz')
 assert np.array_equal(row,saved['rows']) and np.array_equal(h,saved['consistency'])
 ra=np.load(D/'clock_row_preparation_v2_446/row_arrays.npz');soft=ra['keep']&ra['soft']
 theta=max(0.,float(-row[soft].min()),float(abs(h).max()));mu=read(ACCEPTED/'decision.json')['summary']['phi']
 assert np.isfinite(row).all() and np.isfinite(h).all() and np.isfinite(theta) and np.isfinite(mu)
 eligible=bool(theta<mu)
 assert read(O/'ceiling.json')==dict(mu=mu,theta=theta,ceiling=(mu+theta)/2 if eligible else None,eligible=eligible,epigraph=float(w[-1]),epigraph_is_not_physical_permission=True)
 assert decision['accepted']==eligible
 if eligible:assert np.load(O/'witness.npy').tobytes()==w.tobytes()
 else:assert not (O/'witness.npy').exists()
 event('audit_completed',origin)
 write(O/'audit.json',dict(passed=True,candidate_accepted=eligible,full_original_checks=checks,
   backend_point_unchanged=True,backend_duals_used=True,old_zero_face_analytic_dual_used=False,
   strict_modeled_improvement=eligible,theta=theta,mu=mu,finite_maps=0,model_calls=0,physical_admission=False))

if __name__=='__main__':main()
