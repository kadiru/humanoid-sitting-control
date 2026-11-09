"""Preflight ONLY for one complete integer-timed path formulation; zero optimizer/physics.
No completed experiment is repeated. All373 signed failures are preserved and seed only.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from low_seat_discrete_problem_377 import make_discrete
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'discrete_preparation_377'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);d=make_discrete();p=d['p'];x=d['x0'];free=d['free'];eq,iq=d['evaluate'](x);E,I=d['evaluate'](x,True);X=p['last']['X'].copy();Y=p['last']['Y'].copy();src=np.load(D/'support_path_373/final.npz')
 reconstruction=float(max(abs(X-src['full_coordinates']).max(),abs(Y-src['local_outputs']).max()));assert reconstruction<1e-9
 assert np.array_equal(x[:1026],src['x'][:1026]);assert len(free)==786
 rng=np.random.default_rng(377);direction=np.zeros(len(x));direction[free]=rng.normal(size=len(free));direction/=np.linalg.norm(direction);h=1e-5
 ep,ip=d['evaluate'](x+h*direction);em,im=d['evaluate'](x-h*direction)
 fd=np.r_[(ep-em)/(2*h),(ip-im)/(2*h)];an=np.r_[E@direction,I@direction];err=float(abs(fd-an).max());assert err<2e-6
 initial_slack=max(0.,float((-iq[d['keep']&~d['hard']]/d['scale'][d['keep']&~d['hard']]).max()))+1e-6
 np.savez_compressed(O/'fixture.npz',x=x,free=free,bounds=np.array(p['bounds']),eq=eq,ineq=iq,raw_ineq=iq+d['tighten'],E=E[:,free],I=I[:,free],direction=direction,h=h,fd=fd,analytic=an,scale=d['scale'],keep=d['keep'],hard=d['hard'],tighten=d['tighten'],segment_ticks=d['ticks'],ulp=d['ulp'],rate_per_tick_envelope=d['rate_per_tick_envelope'],target_envelope=d['target_envelope'],initial_slack=initial_slack,X=X,Y=Y)
 paths=[Path(__file__).resolve(),Path('scripts/low_seat_discrete_problem_377.py').resolve(),D/'support_path_373/final.npz',D/'support_path_373/results.json',D/'support_path_audit_374/independent_audit.json']
 hashes={str(v):sha(v) for v in paths};old=json.loads((D/'support_path_373/revision.json').read_text())['source_input_sha256']
 for v,hsh in old.items():assert sha(v)==hsh,v;hashes[v]=hsh
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,source_input_sha256=hashes,maximum_reconstruction_error=reconstruction,raw_directional_error=err,unchanged_derivative_bound=2e-6,original_rows=len(iq),retained_rows=int(d['keep'].sum()),hard_rows=int((d['keep']&d['hard']).sum()),soft_rows=int((d['keep']&~d['hard']).sum()),free_path_variables=len(free),declared_path_variables=len(x),movement_ticks=int(d['ticks'].sum()),synthetic_quiet_ticks=25,integer_time_limit_ticks=675,integer_time_s=675*.02,tick_allocation='one grouped Hamilton allocation from373 duration proportions:101 normal +549 other movement; >=one tick per segment, ascending index tie break; one20ms normal construction reserve; no sweep',construction_contact_interior_BW=d['contact_interior'],construction_phase_increment=d['phase_interior'],construction_support_interior_m=d['support_interior'],construction_rate_roundoff_rad_per_tick=d['rate_per_tick_envelope'].tolist(),construction_target_roundoff_rad=d['target_envelope'].tolist(),initial_original_feasibility_not_claimed=True,initial_hard_scaled_minimum=float((iq[d['keep']&d['hard']]/d['scale'][d['keep']&d['hard']]).min()),initial_slack=initial_slack,exact_rounded_filter_is_separate_mandatory_gate=True,all675_ticks_and_intermediate_geometry_must_be_checked=True,original_physical_criteria_unchanged=True,physics_ticks=0,policy_optimizer_updates=0,nominal_optimizer_runs=0,physical_admission=False)
 (O/'preparation.json').write_text(json.dumps(r,indent=2));(O/'SHA256.json').write_text(json.dumps({v.name:sha(v) for v in O.iterdir() if v.is_file()},indent=2));print(json.dumps({k:v for k,v in r.items() if k!='source_input_sha256'},indent=2),flush=True)
if __name__=='__main__':main()
