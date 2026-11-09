"""Independent returned full-coordinate point audit; no solver/producer import."""
from common import *

def main():
 origin=admission('audit');contract()
 assert read(O/'producer_exit.json')['completed_within_deadline']
 proof=read(O/'input_proof.json');assert proof['passed']
 for name,h in proof['file_sha256'].items():assert sha(O/name)==h,name
 marker=read(O/'scipy_started.json');backend=read(O/'backend_result.json')
 assert marker['method']=='highs-ipm' and marker['cold'] and not marker['x0_or_basis']
 assert marker['input_proof_sha256']==sha(O/'input_proof.json')
 assert marker['options']['time_limit']<=CAP-marker['elapsed_s']+0.01
 decision=read(O/'candidate.json')
 if not decision['accepted']:
  assert not (O/'witness.npy').exists()
  write(O/'audit.json',dict(passed=True,candidate_accepted=False,
    scope='Input proof, finite termination and rejection accounting only; no acceptable point inferred',
    backend_status=backend['status'],no_retry=True,physical_admission=False));return
 assert backend['status']==0 and backend['success']
 import numpy as np
 from scipy import sparse
 raw=np.load(O/'backend_primal.npy');w=np.load(O/'witness.npy')
 assert raw.dtype==w.dtype==np.dtype('float64') and w.shape==(78791,)
 assert raw.tobytes()==w.tobytes() and np.isfinite(w).all() and w[-1]==0
 dual=np.load(D/'zero_face_preparation_572/analytic_lower_bound_dual.npz')
 lower=dual['lower'];upper=dual['upper'];checks={}
 assert np.all(upper==0) and np.all(lower[:-1]==0) and lower[-1]==1
 for role in ['full','native']:
  a=dict(np.load(O/(role+'_arrays.npz')));M=sparse.load_npz(O/(role+'_matrix.npz'));E=sparse.load_npz(O/(role+'_equality.npz'))
  row=M.dot(w)-a['rhs'];eq=E.dot(w)-a['equality_rhs'];gradient=a['cost']+upper-lower
  lf=np.isfinite(a['bounds'][:,0]);uf=np.isfinite(a['bounds'][:,1])
  primal=max(0.,float(np.max(row,initial=0)),float(np.max(abs(eq),initial=0)),
    float(np.max(a['bounds'][:,0]-w)),float(np.max(w-a['bounds'][:,1])))
  station=float(np.max(abs(gradient)));gap=abs(float(a['cost'].dot(w)+a['bounds'][uf,1].dot(upper[uf])-a['bounds'][lf,0].dot(lower[lf])))
  finite=all(np.isfinite(x).all() for x in [row,eq,gradient,w,upper,lower,np.asarray([primal,station,gap])])
  signs=bool(np.all(lower>=0) and np.all(upper>=0) and np.all(lower[~lf]==0) and np.all(upper[~uf]==0))
  checks[role]=dict(primal=primal,dual=station,absolute_gap=gap,finite=bool(finite),dual_signs=signs,
    passed=bool(finite and signs and max(primal,station,gap)<=1e-9))
  saved=np.load(O/(role+'_residuals.npz'))
  for key,val in [('inequality',row),('equality',eq),('stationarity',gradient)]:assert val.tobytes()==saved[key].tobytes()
  assert checks[role]['passed'];check(origin)
 assert checks==read(O/'original_checks.json')['systems']
 state=np.load(O/'state.npz');A=sparse.load_npz(O/'A.npz');C=sparse.load_npz(O/'C.npz')
 row=state['r']+A.dot(w[:-1]);h=state['h']+C.dot(w[:-1]);saved=np.load(O/'primary_prediction.npz')
 assert row.tobytes()==saved['rows'].tobytes() and h.tobytes()==saved['consistency'].tobytes()
 ra=np.load(D/'clock_row_preparation_v2_446/row_arrays.npz');soft=ra['keep']&ra['soft']
 theta=max(0.,float(-row[soft].min()),float(abs(h).max()));mu=read(PRIOR/'decision.json')['summary']['phi']
 assert read(O/'ceiling.json')==dict(mu=mu,theta=theta,ceiling=(mu+theta)/2,eligible=theta<mu)
 event('audit_completed',origin)
 write(O/'audit.json',dict(passed=True,candidate_accepted=True,systems=checks,
   returned_point_bitwise_backend=True,original_scalar_already_zero=True,old_reductions_used=False,
   full_original_native_checks=2,finite_maps=0,model_calls=0,physical_admission=False,
   witness_sha256=sha(O/'witness.npy'),actual_ceiling=read(O/'ceiling.json')))

if __name__=='__main__':main()
