"""One new-state full primary with two actual-gap cuts; no warm start or fallback."""
from common import *

def main():
 origin=admission('producer');spec=contract()
 for path,h in {**spec['input_sha256'],**spec['runtime_sha256']}.items():assert sha(path)==h,path;check(origin)
 for folder in ['lifted_feasibility_audit_491','torque_path_audit_525']:
  assert read(D/folder/'independent_audit.json')['passed']
 assert read(PRIOR/'audit.json')['passed'] and read(PRIOR/'result.json')['trial_merit_accepted']
 event('input_identity_verified',origin)
 import numpy as np
 from scipy import sparse
 from builder import load,intervals,assemble
 d,f,y,J,drops,b=load();s=intervals(d,f,drops,1.0);assert all(max(abs(q.numerator).bit_length(),q.denominator.bit_length())<=2304 for k in ['lower','upper','widths','closed_lower','closed_upper'] for q in s[k]);check(origin)
 p=assemble(d,b,s,None)
 assert p['matrix'].nnz<=12000000 and b['C'].nnz<=6000000;check(origin)
 np.savez(O/'state.npz',features=f,actualY=y,J=J,**{k:b[k] for k in ['r','e','h','guard']})
 for name,value in [('A',b['A']),('C',b['C']),('H',s['H'])]:sparse.save_npz(O/(name+'.npz'),value)
 write(O/'intervals.json',{**{key:[[x.numerator,x.denominator] for x in s[key]] for key in ['lower','upper','widths','closed_lower','closed_upper']},'owners':s['owners'],'drops':[[x.numerator,x.denominator] for x in drops],'trust':1.0})
 sparse.save_npz(O/'full_matrix.npz',p['matrix']);sparse.save_npz(O/'full_equality.npz',p['equality'])
 a={k:p[k] for k in ['rhs','cost','bounds','equality_rhs']};np.savez(O/'full_arrays.npz',**a)
 from independent_input import prove
 prove(origin)
 from target_rows import build
 actual=np.load(PRIOR/'actual.npz')['actual_rows'];G,grhs=build(d['P'],J,actual,d['rows']['scale'])
 from independent_targets import prove_targets,prove_augmented
 proof=prove_targets(d['P'],J,actual,d['rows']['scale'],G,grhs)
 sparse.save_npz(O/'target_matrix.npz',G);np.save(O/'target_rhs.npy',grhs)
 M=sparse.vstack([p['matrix'],G],format='csr');aa=dict(a,rhs=np.r_[a['rhs'],grhs]);E=p['equality']
 assert M.shape==(652531,78791) and G.nnz<=157580
 proof.update(prove_augmented(M,aa,p['matrix'],a,G,grhs))
 sparse.save_npz(O/'augmented_matrix.npz',M);np.savez(O/'augmented_arrays.npz',**aa)
 write(O/'target_input_proof.json',dict(proof,source_J_sha256=sha(PRIOR/'accepted_J.npz'),source_attribution_sha256=sha(ATTR/'attribution.json'),full_matrix_sha256=sha(O/'full_matrix.npz'),target_matrix_sha256=sha(O/'target_matrix.npz'),target_rhs_sha256=sha(O/'target_rhs.npy'),augmented_matrix_sha256=sha(O/'augmented_matrix.npz'),augmented_arrays_sha256=sha(O/'augmented_arrays.npz'),augmented_nnz=int(M.nnz)))
 event('new_full_input_and_target_proof_completed',origin)
 from scipy.optimize import linprog
 remaining=SOLVE_END-(time.monotonic()-origin)
 if remaining<=0:
  write(O/'candidate.json',dict(accepted=False,reason='No remaining solver allowance after charged assembly/proof',no_retry=True));return
 options=dict(time_limit=remaining,maxiter=4096,presolve=True,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9,ipm_optimality_tolerance=1e-9)
 write(O/'scipy_started.json',dict(method='highs-ipm',options=options,cold=True,x0_or_basis=False,elapsed_s=time.monotonic()-origin))
 result=linprog(a['cost'],A_ub=M,b_ub=aa['rhs'],A_eq=E,b_eq=a['equality_rhs'],bounds=a['bounds'],method='highs-ipm',options=options)
 write(O/'backend_result.json',dict(status=int(result.status),success=bool(result.success),message=str(result.message),candidate_returned=result.x is not None,iterations=int(result.nit),crossover_iterations=int(getattr(result,'crossover_nit',0)),total_elapsed_s=time.monotonic()-origin));check(origin)
 if not(result.status==0 and result.success and result.x is not None):
  write(O/'candidate.json',dict(accepted=False,reason='No successful complete backend return',no_retry=True));return
 w=np.asarray(result.x);np.save(O/'backend_primal.npy',w)
 assert w.dtype==np.float64 and w.shape==(78791,) and np.isfinite(w).all()
 m=dict(inequality=np.asarray(result.ineqlin.marginals),equality=np.asarray(result.eqlin.marginals),upper=np.asarray(result.upper.marginals),lower=np.asarray(result.lower.marginals))
 np.savez(O/'backend_marginals.npz',**m)
 from numerical_checks import check_point,added_duals_zero
 aug,ares=check_point(M,E,aa,w,m)
 # The exact-zero gate makes prefix certificate extraction a literal unchanged subvector.
 zero=added_duals_zero(m['inequality'][-2:])
 base_m=dict(m,inequality=m['inequality'][:-2])
 original,ores=check_point(p['matrix'],E,a,w,base_m)
 np.savez(O/'augmented_residuals.npz',**ares);np.savez(O/'original_residuals.npz',**ores)
 target_residual=grhs-G@w;np.save(O/'target_prediction.npy',target_residual)
 target_pass=bool(np.all(target_residual>=0))
 row=b['r']+b['A']@w[:-1];h=b['h']+b['C']@w[:-1];soft=d['rows']['keep']&d['rows']['soft']
 theta=max(0.,float(-row[soft].min()),float(abs(h).max()));mu=float(b['phi'])
 from fractions import Fraction as F
 useful=bool(np.isfinite(theta) and theta>=0 and 100*(F(mu)-F(theta))>=F(mu))
 np.savez(O/'primary_prediction.npz',rows=row,consistency=h)
 checks=dict(augmented=aug,original=original,added_marginals_exact_zero=zero,target_prediction_pass=target_pass,one_percent_improvement=useful,mu=mu,theta=theta)
 write(O/'checks.json',checks)
 accepted=bool(aug['passed'] and original['passed'] and zero and target_pass and useful)
 write(O/'candidate.json',dict(accepted=accepted,reason='All new-input, original and augmented numerical gates passed' if accepted else 'At least one original, target or certificate gate failed',no_retry=True))
 if accepted:np.save(O/'witness.npy',w)
 event('producer_completed',origin)
if __name__=='__main__':main()
