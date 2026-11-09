"""Saved-array preparation for a new integer-exact all-hard closest-path problem.
No FK/root/model/derivative fixture/path optimizer/schedule execution.
"""
from pathlib import Path
import numpy as np,json,hashlib,datetime
from low_seat_lattice_projection_428 import *
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'lattice_preparation_428'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False)
 a=json.loads((D/'budget_path_audit_425/independent_audit.json').read_text());assert a['passed']
 f=dict(np.load(D/'monotone_preparation_410/fixture.npz'));b=dict(np.load(D/'window_path_400/initial.npz'));s=dict(np.load(D/'budget_path_424/structural.npz'));e=dict(np.load(D/'budget_path_424/event_clock.npz'))
 parts,req,phase=necessary(s['C'],s['Y'][:,54],b['rates'],b['command_lo'],b['command_hi'],e['ulp']);budgets,proof=allocate(phase)
 reference=s['z'][:894].copy();radius=np.r_[np.ones(785),np.full(109,.1)];lower=f['z'][:894]-radius;upper=f['z'][:894]+radius
 assert np.all(reference>=lower) and np.all(reference<=upper)
 base=np.load(D/'direct_preparation_416/fixture.npz');lo=base['lo'];hi=base['hi'];keep=f['keep'];scale=f['scale']
 # Only algebra of the new hard rows/objective is evaluated on the EXISTING410
 # derivative fixture. The integer initialization is not falsely called feasible.
 values,J=rows(f['iq'],f['v'],scale,keep,lo,hi,f['I'],f['G']);rng=np.random.default_rng(428);direction=rng.normal(size=894);direction/=np.linalg.norm(direction)
 independent=np.r_[(f['I'][:,:894]@direction)[keep]/scale[keep],f['G'][:786,:894]@direction,-f['G'][:786,:894]@direction]
 error=float(abs(J@direction-independent).max());assert error<2e-12
 test=reference+.001*radius*direction;val,grad=objective(test,reference,radius);h=1e-6;fd=(objective(test+h*direction,reference,radius)[0]-objective(test-h*direction,reference,radius)[0])/(2*h);oe=abs(fd-grad@direction);assert oe<2e-10
 np.savez_compressed(O/'fixture.npz',reference=reference,budgets=budgets,seed=np.r_[reference,budgets],lower=lower,upper=upper,radius=radius,lo=lo,hi=hi,keep=keep,scale=scale,values=values,jacobian=J,direction=direction,independent_direction=independent,objective_test=test,objective_gradient=grad,objective_value=val,objective_fd=fd,parts=parts,required=req,phase_required=phase,root_commands=s['C'],root_gaps=s['Y'][:,54],ulp=e['ulp'])
 (O/'allocation_proof.json').write_text(json.dumps(proof,indent=2))
 prior=json.loads((D/'monotone_preparation_410/preparation.json').read_text());hashes=prior['source_input_sha256'].copy()
 for p in [Path(__file__),Path('scripts/low_seat_lattice_projection_428.py'),D/'budget_path_424/structural.npz',D/'budget_path_424/results.json',D/'budget_path_424/event_clock.npz',D/'budget_path_audit_425/independent_audit.json',D/'monotone_preparation_410/fixture.npz',D/'direct_preparation_416/fixture.npz',D/'window_path_400/initial.npz']:hashes[str(p.resolve())]=sha(p)
 for p,h in hashes.items():assert sha(p)==h,p
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,reference='exported424root FINAL only; explicitly a new research input, not admitted rounding or continued tree',wholepath_required_phase_seconds=phase.tolist(),conditional_fixed_waveform_event_tick_ceiling=[int(np.ceil(t/.02)) for t in phase],budgets=budgets.tolist(),allocation_objective=proof['objective'],allocation_scope='Exact finite arithmetic minimax necessary-duration/allocated-duration resource problem. Optimum is not a nonlinear or physical feasibility certificate.',new_nonlinear_problem='All28887 retained model/window/coordinate rows HARD; minimize normalized squared complete-path displacement, ZERO slack variables; one30-iteration SLSQP call; no fallback or callback selection.',buffer_dimensionless=BUFFER,continuous_domain='Exact416 box centered406 retained, not recentered at424',derivative_algebra_error=error,objective_gradient_error=oe,full_model_derivative_410=prior['full_directional_error'],unchanged_derivative_bound=2e-6,initial_integer_path_evaluated=False,optimizer_calls=0,new_model_evaluations=0,physics_ticks=0,physical_admission=False,source_input_sha256=hashes)
 (O/'preparation.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps({k:v for k,v in result.items() if k!='source_input_sha256'},indent=2))
if __name__=='__main__':main()
