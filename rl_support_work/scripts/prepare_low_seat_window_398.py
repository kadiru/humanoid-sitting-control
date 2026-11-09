"""Preflight of a NEW complete-path continuous-window formulation. No optimizer.
The saved394 defect motivates actual nonlinear acceptance, not tangent admission.
The seed is saved388; no old optimization, projection study, or audit is repeated.
"""
from pathlib import Path
import numpy as np,json,hashlib,datetime
from low_seat_window_envelope_398 import make_window
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'window_preparation_398'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    O.mkdir(exist_ok=False);d=make_window();v=d['initial'].copy();eq,iq=d['evaluate'](v);E,I=d['evaluate'](v,True)
    s=d['last'];env=s['env'];save=dict(v=v,x=s['x'],X=s['X'],Y=s['Y'],C=s['C'],dt=s['dt'],T=s['T'],B=s['B'],eq=eq,ineq=iq,raw=s['raw'],tighten=s['tighten'],E=E,I=I,scale=d['scale'],keep=d['keep'],ulp=d['ulp'],lo=d['p']['command_lo'],hi=d['p']['command_hi'],rates=d['p']['rates'])
    save.update({'env_'+k:val for k,val in env.items()})
    seed=dict(np.load(D/'event_path_388/final.npz'))
    reconstruct=float(max(abs(s['X']-seed['full_coordinates']).max(),abs(s['Y']-seed['local_outputs']).max(),abs(iq[:20284]-seed['ineq'][:20284]).max()))
    rng=np.random.default_rng(398);direction=rng.normal(size=900);direction[-5:]-=direction[-5:].mean();direction/=np.linalg.norm(direction);h=1e-5
    ep,ip=d['evaluate'](v+h*direction);em,im=d['evaluate'](v-h*direction)
    fd=np.r_[(ep-em)/(2*h),(ip-im)/(2*h)];an=np.r_[E@direction,I@direction]
    err=float(abs(fd-an).max());save.update(direction=direction,h=h,fd=fd,analytic=an)
    np.savez_compressed(O/'fixture.npz',**save)
    hashes={str(p.resolve()):sha(p) for p in [Path(__file__),Path('scripts/low_seat_window_envelope_398.py'),Path('scripts/low_seat_event_clock_v2_386.py'),D/'event_path_388/final.npz',D/'event_path_388/event_clock.npz',D/'integer_restoration_defect_397.json']}
    hashes.update(json.loads((D/'event_path_388/revision.json').read_text())['source_input_sha256'])
    for p,hv in hashes.items():assert sha(p)==hv,p
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=bool(err<2e-6 and reconstruct<1e-10),reconstruction_error=reconstruct,full_directional_error=err,original_directional_error=float(abs(fd[1:20285]-an[1:20285]).max()),window_directional_error=float(abs(fd[20285:]-an[20285:]).max()),unchanged_derivative_bound=2e-6,source_input_sha256=hashes,rows=len(iq),window_rows=7424,breakpoints=232,variables=900,scope=__doc__,actual_float32_schedule=False,physical_admission=False,physics_ticks=0,policy_optimizer_updates=0)
    (O/'preparation.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='source_input_sha256'},indent=2));assert result['passed']
if __name__=='__main__':main()
