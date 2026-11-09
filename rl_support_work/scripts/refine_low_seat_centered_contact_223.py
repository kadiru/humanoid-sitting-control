"""Bounded offline contact-consistent centered-path refinement. No physics.
Only root-x and pitch vary for cases0/6 under the prior bounds. Centered y/roll/yaw,
full fixed-foot constraints, arms, staged path, all clearance/force/command criteria
remain unchanged. Max40 unique19-knot checks/10iterations plus ONE91-knot dense
check per geometry. Candidate bounds do not establish dynamic feasibility.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.optimize import minimize
from humanoid_sitting.shared_frame import BodyModel
from low_seat_centered_coarse_223 import build_plan as coarse
from low_seat_fixed_foot_plan_126 import build_plan as dense
from low_seat_balanced_force_v2_145 import reallocate
from low_seat_touchdown_path_175 import TouchdownPath
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'centered_contact_refinement_223'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Budget(Exception):pass

def main():
 O.mkdir(exist_ok=False);mp=D/'cooked_model_50/model.json';ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';cp=D.parent/'contact_effort_repair_01/plane_calibration.json';m=BodyModel(mp);meta=read(mp);ine=read(ip);cal=read(cp)
 sources=[Path(__file__).resolve(),mp,ip,cp]+[Path(__file__).with_name(x) for x in ['low_seat_centered_coarse_223.py','low_seat_fixed_foot_plan_126.py','low_seat_balanced_force_v2_145.py','low_seat_touchdown_path_175.py']]+[D/'centered_path_check_221'/f'case_{c}.json' for c in [0,6]]
 a=Path(__file__).with_name('low_seat_fixed_foot_plan_126.py').read_text();b=Path(__file__).with_name('low_seat_centered_coarse_223.py').read_text().split('\n',1)[1];assert b==a.replace('for k in range(91):','for k in range(0,91,5):')
 revision=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,maximum_evaluations_per_geometry=40,maximum_iterations=10,dense_checks_per_geometry=1,root_x_bounds_m=[-.28,-.1],pitch_bounds_deg=[5,38],source_sha256={str(p.resolve()):sha(p) for p in sources},coarse_transform_exact=True,physics_ticks=0,optimizer_updates=0,holdout_used=False)
 (O/'revision.json').write_text(json.dumps(revision,indent=2));rows=[]
 for c in [0,6]:
  old=read(D/'centered_path_check_221'/f'case_{c}.json');x=old['inputs'];history=[];cache={}
  def build(z,fn):
   p=fn(m,meta,ine,cal,x['q'],x['gravity'],x['geometry'],x['reference'],x['command'],[float(z[0]),0.,0.,float(z[1]),0.]);return reallocate(m,ine,p,x)
  def evaluate(z):
   key=tuple(float(v) for v in z)
   if key in cache:return cache[key]
   if len(cache)>=40:raise Budget()
   p=build(z,coarse);path=p['path'];other=all(e['other_model_passed'] and e['load_checks'][0]['passed'] for e in path)
   constraints=np.array([min(e['foot_cop_margin_m'] for e in path)-.005,min(e['patch_margin_m'] for e in path if e['pelvis_gap_m']<.02)-.015,min(e['floor_clearance_m'] for e in path)-.005,.003-max(e['foot_position_error_m'] for e in path),np.deg2rad(.5)-max(e['foot_orientation_error_rad'] for e in path),min(e['load_checks'][0]['min_command_margin_rad'] for e in path)-.03,1-max(e['load_checks'][0]['maximum_effort_utilization'] for e in path),min(v['margin_m'] for e in path for v in e['load_checks'][0]['individual_foot_cop'].values())-.005])
   rec=dict(index=len(history),parameters=list(key),clearance_m=min(e['clearance_m'] for e in path),other_passed=other,constraints=constraints.tolist(),summary=p['summary']);history.append(rec);cache[key]=rec;(O/f'coarse_{c}_{rec["index"]:02d}.json').write_text(json.dumps(dict(case=c,inputs=x,parameters=list(key),plan=p,record=rec),indent=2));print('COARSE',c,rec['index'],rec['clearance_m'],other,flush=True);return rec
  start=np.array(old['endpoint_parameters'])[[0,3]];result=None;budget=False
  try:
   result=minimize(lambda z:-100*evaluate(z)['clearance_m'],start,method='SLSQP',bounds=[(-.28,-.1),(np.deg2rad(5),np.deg2rad(38))],constraints=[dict(type='ineq',fun=lambda z:100*np.array(evaluate(z)['constraints']))],options=dict(maxiter=10,eps=1e-5,ftol=1e-9))
  except Budget:budget=True
  valid=[z for z in history if z['other_passed']];best=max(valid,key=lambda z:z['clearance_m']) if valid else max(history,key=lambda z:min(z['constraints']))
  p=build(best['parameters'],dense);schedule=dict(checked=False,physical_admission=False)
  if p['model_passed']:
   engine=TouchdownPath(p,m);targets=[];nom=[]
   for tick in range(1,1501):
    engine.settle=dict(quiet=True);targets.append(engine.advance(True,.1).copy());nom.append(engine.nominal.copy())
    if engine.position>=engine.arc[-1]-1e-10:break
   schedule=dict(checked=True,synthetic_quiet_and_load=True,duration_s=tick*.02,ticks=tick,complete=bool(engine.position>=engine.arc[-1]-1e-10),time_gate_passed=bool(engine.position>=engine.arc[-1]-1e-10 and tick*.02<=13.5),physical_admission=False);np.savez_compressed(O/f'schedule_{c}.npz',targets=np.array(targets),nominal=np.array(nom),initial=np.array(x['command'],np.float32))
  summary=dict(case=c,evaluations=len(history),budget_reached=budget,optimizer_message=str(result.message) if result is not None else 'bounded evaluation cap',best_coarse=best,dense_summary=p['summary'],model_passed=p['model_passed'],schedule=schedule)
  (O/f'case_{c}.json').write_text(json.dumps(dict(case=c,inputs=x,endpoint_parameters=[best['parameters'][0],0.,0.,best['parameters'][1],0.],plan=p,history=history,summary=summary,schedule=schedule),indent=2));rows.append(summary);print('DENSE',json.dumps(summary),flush=True)
 (O/'results.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,scope=__doc__,source_sha256=revision['source_sha256'],physics_ticks=0,optimizer_updates=0,physical_admission=False),indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()
