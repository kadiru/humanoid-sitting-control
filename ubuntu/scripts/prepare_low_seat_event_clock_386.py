"""Preflight for event-only clock alignment386, no optimizer or physics.
379 event budgets are reused exactly; interior durations are newly free.
All675 ideal ticks participate in constraints; actualfloat32 replay remains a later gate.
"""
from pathlib import Path
import json,hashlib,datetime,numpy as np
from low_seat_event_clock_386 import make_event,GROUPS
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'event_clock_preparation_386'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);d=make_event();z=d['z0'];eq,iq=d['evaluate'](z);E,I=d['evaluate'](z,True);saved=dict(d['last']);src=np.load(D/'support_path_373/final.npz');reconstruction=float(max(abs(saved['X']-src['full_coordinates']).max(),abs(saved['Y']-src['local_outputs']).max()))
 rng=np.random.default_rng(386);direction=rng.normal(size=len(z));direction/=np.linalg.norm(direction);h=1e-5
 ep,ip=d['evaluate'](z+h*direction);active_plus=d['last']['active'].copy();em,im=d['evaluate'](z-h*direction);active_minus=d['last']['active'].copy();fd=np.r_[(ep-em)/(2*h),(ip-im)/(2*h)];an=np.r_[E@direction,I@direction];error=float(abs(fd-an).max())
 slack=max(0.,float((-iq[d['keep']&~d['hard']]/d['scale'][d['keep']&~d['hard']]).max()))+1e-6
 np.savez_compressed(O/'fixture.npz',z=z,x=saved['x'],T=saved['T'],shape_free=d['shape_free'],budgets=d['budgets'],groups=GROUPS,eq=eq,ineq=iq,raw_ineq=saved['raw'],tighten=saved['tighten'],E=E,I=I,direction=direction,h=h,fd=fd,analytic=an,scale=d['scale'],keep=d['keep'],hard=d['hard'],initial_slack=slack,ulp=d['ulp'],C=saved['C'],clock=saved['samples'],locations=saved['locations'],clock_raw=saved['clock_raw'],clock_full=saved['clock_full'],active=saved['active'],active_plus=active_plus,active_minus=active_minus,targets=saved['targets'],X=saved['X'],Y=saved['Y'])
 checks=dict(reconstruction=reconstruction<1e-9,directional=error<2e-6,fixed_event_budgets=np.array_equal(d['budgets'],[93,194,246,101,16]),finite=bool(np.isfinite(iq).all() and np.isfinite(I).all()),all675_ticks=saved['clock_raw'].shape==(675,64),positive_duration=bool(np.all(saved['x'][1026:]>.02)))
 paths=[Path(__file__).resolve(),Path('scripts/low_seat_event_clock_386.py').resolve(),D/'support_path_373/final.npz',D/'discrete_path_379/results.json',D/'support_path_audit_374/independent_audit.json']
 hashes={str(v):sha(v) for v in paths}
 for path,hs in json.loads((D/'support_path_373/revision.json').read_text())['source_input_sha256'].items():assert sha(path)==hs;hashes[path]=hs
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=bool(all(checks.values())),checks={k:bool(v) for k,v in checks.items()},scope=__doc__,source_input_sha256=hashes,reconstruction_error=reconstruction,directional_error=error,unchanged_derivative_bound=2e-6,event_knots=[0,16,64,99,104,114],event_budgets=d['budgets'].tolist(),movement_ticks=650,quiet_ticks=25,total_ticks=675,free_path_variables=len(z),shape_variables=786,logratio_variables=109,original_support_rows=20284,full_clock_rows=43200,compressed_min_clock_rows=64,active_rows_changed_in_derivative_fixture=int(np.count_nonzero(active_plus!=active_minus)),maximum_duration_group_sum_error_s=float(max(abs(saved['x'][1026+a:1026+b].sum()-budget*.02) for (a,b),budget in zip(GROUPS,d['budgets']))),minimum_dt_s=float(saved['x'][1026:].min()),maximum_timing_J_group_sum=float(max(abs(saved['T'][a:b].sum(0)).max() for a,b in GROUPS)),initial_slack=slack,initial_hard_minimum=float((iq[d['keep']&d['hard']]/d['scale'][d['keep']&d['hard']]).min()),clock_derivative_is_piecewise_active_row=True,original_physical_criteria_unchanged=True,actual_float32_schedule_checked=False,intermediate_geometry_checked=False,physics_ticks=0,policy_optimizer_updates=0,nominal_optimizer_runs=0,physical_admission=False)
 (O/'preparation.json').write_text(json.dumps(r,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps({k:v for k,v in r.items() if k!='source_input_sha256'},indent=2),flush=True);assert r['passed'],checks
if __name__=='__main__':main()
