"""Reconstruct the complete227 centered path with the live-input148 planner.
Two saved development placements only; fixture inputs never enter runtime.
No optimization, physics, relaxed criteria or automatic physical launch.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from humanoid_sitting.shared_frame import BodyModel
from low_seat_balanced_runtime_v2_148 import build_plan
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'centered_runtime_paths_231'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False)
 src=D/'centered_orientation_refinement_v2_227/case_0.json';original=read(src);endpoint=original['endpoint_parameters']
 assert original['summary']['research_pass'] and read(D/'centered_orientation_audit_v2_228/independent_audit.json')['passed']
 config=read(D/'bias_runtime_preparation_v3_209/candidate_config.json');config['routes']=config['routes'][:1];config['routes'][0]['endpoint']=endpoint
 config['scope']='One measured development geometry route, centered complete path227; current-input planner148 and unchanged210 bias/settling. Other geometries retain guarded-normal3s; no physical admission.'
 (O/'candidate_config.json').write_text(json.dumps(config,indent=2))
 mp=Path(config['cooked_model']);ip=Path(config['inertia']);cp=D.parent/'contact_effort_repair_01/plane_calibration.json';m=BodyModel(mp);meta=read(mp);ine=read(ip);cal=read(cp)
 P=D/'bias_probe_210/candidate';tr=np.load(P/'trace.npz');ss=np.load(P/'sensor_inputs.npz')['sensors'];at=int(np.argmin(abs(tr['t']-1)));assert tr['t'][at]==1.
 rows=[]
 for c in [0,1]:
  x=dict(q=tr['q'][at,c].astype(float).tolist(),gravity=ss[0,10,c,339:342].tolist(),geometry=tr['geometry'][at,c].tolist(),reference=[float(tr['estimated_root_x'][at,c]),0.,0.],command=tr['command'][at,c].astype(float).tolist())
  assert np.array_equal(np.array(x['q'][:23],np.float32),ss[0,10,c,270:293])
  # Use the runtime entrypoint, including145's explicit input normalization.
  p=build_plan(m,meta,ine,cal,np.array(x['q'],np.float32),np.array(x['gravity'],np.float32),np.array(x['geometry'],np.float32),x['reference'],np.array(x['command'],np.float32),endpoint)
  if c==0:assert x==original['inputs'];assert p==original['plan']
  p['summary']['case']=c
  loads=[e['load_checks'][0] for e in p['path']];rollids=[m.names.index(s+'_ankle_roll_joint') for s in ['left','right']]
  diff=max(abs(sum(w for w,s in zip(l['corner_fractions'],l['corner_sides']) if s=='left')-sum(w for w,s in zip(l['corner_fractions'],l['corner_sides']) if s=='right')) for l in loads)
  roll=max(abs(np.array(l['command'])[rollids]-np.array(x['command'])[rollids]).max() for l in loads)
  assert p['model_passed'] and diff<=.1 and roll<=.04
  route=config['routes'][0];assert route['height_m'][0]<=p['height']<=route['height_m'][1] and route['slope_deg'][0]<=p['slope_deg']<=route['slope_deg'][1]
  record=dict(case=c,inputs=x,endpoint_parameters=endpoint,plan=p)
  (O/f'case_{c}.json').write_text(json.dumps(record,indent=2));rows.append(dict(case=c,model_passed=p['model_passed'],summary=p['summary'],maximum_assumed_foot_difference_BW=float(diff),maximum_ankle_roll_command_change_rad=float(roll)))
  print(json.dumps(rows[-1]),flush=True)
 paths=[Path(__file__).resolve(),src,D/'centered_orientation_audit_v2_228/independent_audit.json',mp,ip,cp,P/'trace.npz',P/'sensor_inputs.npz',O/'candidate_config.json']+[Path(__file__).with_name(n) for n in ['low_seat_balanced_runtime_v2_148.py','low_seat_balanced_force_v2_145.py','low_seat_fixed_foot_plan_126.py','low_seat_bias_runtime_v2_208.py','low_seat_bias_feedback_v2_207.py','low_seat_touchdown_path_175.py']]
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,rows=rows,case0_original227_exact=True,physics_ticks=0,optimizer_updates=0,physical_admission=False,source_sha256={str(p.resolve()):sha(p) for p in paths})
 (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()
