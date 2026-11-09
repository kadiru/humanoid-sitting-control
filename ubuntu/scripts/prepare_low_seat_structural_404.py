"""ONE new structural-coordinate preflight; no path optimizer or physics."""
from pathlib import Path
import numpy as np,json,hashlib,datetime
from low_seat_structural_coordinates_404 import make_structural,FLOOR
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'structural_preparation_404'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    O.mkdir(exist_ok=False);d=make_structural();z=d['initial'].copy();eq,iq=d['evaluate'](z);E,I=d['evaluate'](z,True)
    s=d['last'];p=d['p'];save={k:v for k,v in s.items() if isinstance(v,np.ndarray)}
    save.update(initial=z,floor=FLOOR,active=np.any(p['null']!=0,axis=2)|np.any(p['wcom']!=0,axis=2),null=p['null'],wcom=p['wcom'],wbase=p['wbase'],PP=p['PP'],shape=d['shape'],scale=d['d']['scale'],keep=d['d']['keep'],seed_X=dict(np.load(D/'discrete_path_379/final.npz'))['full_coordinates'])
    save.update({f'env_{k}':v for k,v in s['env'].items() if isinstance(v,np.ndarray)})
    save['coordinate_slices']=np.array([[v.start,v.stop] for v in d['slices']]+[[d[k].start,d[k].stop] for k in ['orientation_slice','descent_slice','timing_slice','budget_slice']])
    # Fixture is tangent to the mandatory sum650 budget identity; it is not an allocation trial.
    rng=np.random.default_rng(404);direction=rng.normal(size=len(z));direction[-5:]-=direction[-5:].mean();direction/=np.linalg.norm(direction);h=1e-5
    ep,ip=d['evaluate'](z+h*direction);vp=d['last']['v'].copy();yp=d['last']['Y'].copy()
    em,im=d['evaluate'](z-h*direction);vm=d['last']['v'].copy();ym=d['last']['Y'].copy()
    fd=np.r_[(ep-em)/(2*h),(ip-im)/(2*h)];analytic=np.r_[E@direction,I@direction]
    error=float(abs(fd-analytic).max());maperror=float(abs((vp-vm)/(2*h)-save['G']@direction).max())
    reconstruction=float(abs(save['loads']-save['Y'][:,46:54]).max());gaperr=float(abs(save['Y'][98:103,54]-save['targets']).max())
    active=save['active'];positive=float(save['Y'][:,46:54][active].min());inactive=bool(np.all(save['Y'][:,46:54][~active]==0))
    save.update(direction=direction,h=h,fd=fd,analytic=analytic,plus_v=vp,minus_v=vm,plus_Y=yp,minus_Y=ym)
    np.savez_compressed(O/'fixture.npz',**save)
    inputs=[Path(__file__),Path('scripts/low_seat_structural_coordinates_404.py'),D/'discrete_path_379/final.npz',D/'window_path_400/results.json',D/'window_zero_step_witness_403.json']
    hashes=json.loads((D/'window_path_400/revision.json').read_text())['source_input_sha256'];hashes.update({str(p.resolve()):sha(p) for p in inputs})
    for path,hv in hashes.items():assert sha(path)==hv,path
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,passed=bool(error<2e-6 and maperror<2e-6 and reconstruction<1e-10 and gaperr<1e-10 and positive>0 and inactive),variables=len(z),original_and_window_rows=len(iq),full_directional_error=error,coordinate_map_directional_error=maperror,unchanged_derivative_bound=2e-6,positive_load_reconstruction_error=reconstruction,minimum_active_corner_fraction=positive,inactive_corner_fractions_exact_zero=inactive,five_gap_equation_error=gaperr,new_construction_floor=FLOOR,orientation_minimum_increment=float(save['orientation_increments'].min()),prehover_minimum_increment=float(save['prehover_increments'].min()),posthover_phase_order_is_still_an_explicit_constraint=True,remaining_constraints_may_fail=True,minimum_original_inward_and_window_row=float(iq.min()),maximum_hover_equation_residual=float(abs(eq).max()),source_input_sha256=hashes,actual_float32_schedule=False,intermediate_geometry_checked=False,physical_admission=False,physics_ticks=0,policy_optimizer_updates=0)
    (O/'preparation.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps({k:v for k,v in result.items() if k!='source_input_sha256'},indent=2));assert result['passed']
if __name__=='__main__':main()
