"""Bounded offline centered-path check. No optimization, runtime or physical launch.
Use prior cooked50 sagittal endpoints, same120 staging/inward1mm adjustment,
126 full fixed-foot IK,145v2 smooth force allocation and175 unchanged follower.
Four unique geometries only; every failed model/command/time gate is retained.
"""
from pathlib import Path
import json,datetime,hashlib
import numpy as np
from humanoid_sitting.shared_frame import BodyModel
from low_seat_fixed_foot_plan_126 import build_plan
from low_seat_balanced_force_v2_145 import reallocate
from low_seat_touchdown_path_175 import TouchdownPath
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'centered_path_check_221'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);mp=D/'cooked_model_50/model.json';ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';cp=D.parent/'contact_effort_repair_01/plane_calibration.json';src=D/'terminal_cooked_screen_50/results.json';m=BodyModel(mp);meta=read(mp);ine=read(ip);cal=read(cp);endpoints=read(src)['rows'];P=D/'normal_landing3s_06/candidate';tr=np.load(P/'trace.npz');ss=np.load(P/'sensor_inputs.npz')['sensors'];at=int(np.argmin(abs(tr['t']-1)));rows=[]
 sources=[Path(__file__).resolve(),mp,ip,cp,src,P/'trace.npz',P/'sensor_inputs.npz']+[Path(__file__).with_name(x) for x in ['low_seat_fixed_foot_plan_126.py','low_seat_balanced_force_v2_145.py','low_seat_touchdown_path_175.py','low_seat_gated_command_128.py']]
 rev=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,cases=[0,2,6,10],maximum_path_builds=4,knots_each=91,optimization_evaluations=0,maximum_command_ticks_each=1500,unchanged_criteria=True,no_automatic_physics=True,source_sha256={str(p.resolve()):sha(p) for p in sources})
 (O/'revision.json').write_text(json.dumps(rev,indent=2))
 for c in [0,2,6,10]:
  old=next(z for z in endpoints if z['case']==c and z['assumption']=='foot_only_before_contact')['best_other_screens_feasible'];par=[old['x']-.001,0.,0.,old['pitch_rad'],0.]
  q=tr['q'][at,c].astype(float);x=dict(q=q.tolist(),gravity=ss[0,10,c,339:342].tolist(),geometry=tr['geometry'][at,c].tolist(),reference=[float(tr['estimated_root_x'][at,c]),0.,0.],command=tr['command'][at,c].astype(float).tolist());assert np.array_equal(q[:23].astype(np.float32),ss[0,10,c,270:293])
  p=build_plan(m,meta,ine,cal,q,x['gravity'],x['geometry'],x['reference'],x['command'],par)
  original=p;p=reallocate(m,ine,p,x);p['summary']['case']=c
  schedule=dict(checked=False,physical_admission=False)
  if p['model_passed']:
   e=TouchdownPath(p,m);seq=[];nom=[]
   for tick in range(1,1501):
    # Explicit synthetic quiet/current support-permission fixtures only.
    # They are NOT sensed support and are never used by a runtime.
    e.settle=dict(quiet=True);target=e.advance(True,.1);seq.append(target.copy());nom.append(e.nominal.copy())
    if e.position>=e.arc[-1]-1e-10:break
   schedule=dict(checked=True,synthetic_quiet_and_load=True,ticks=tick,duration_s=tick*.02,complete=bool(e.position>=e.arc[-1]-1e-10),time_gate_passed=bool(e.position>=e.arc[-1]-1e-10 and tick*.02<=13.5),physical_admission=False)
   np.savez_compressed(O/f'schedule_{c}.npz',targets=np.array(seq),nominal=np.array(nom),initial=np.array(x['command'],np.float32))
  (O/f'case_{c}.json').write_text(json.dumps(dict(case=c,inputs=x,endpoint_parameters=par,plan=p,original_summary=original['summary'],schedule=schedule),indent=2))
  rows.append(dict(case=c,endpoint_parameters=par,summary=p['summary'],model_passed=p['model_passed'],schedule=schedule));print(json.dumps(rows[-1]),flush=True)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,physics_ticks=0,optimizer_updates=0,holdout_used=False,physical_admission=False,rows=rows,source_sha256=rev['source_sha256'])
 (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()
