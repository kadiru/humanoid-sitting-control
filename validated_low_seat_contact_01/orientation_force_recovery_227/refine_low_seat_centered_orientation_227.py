"""Bounded OFFLINE centered-stance contact planning, after219–226.
Fixed y=0. Vary root-x/roll/pitch/yaw ONLY within previously documented58 bounds.
Maximize minimum whole-path housing clearance while enforcing existing gates plus
separate research limits on assumed L/R foot-load difference<=.1BW and ankle-roll
command change<=.04rad. This targets the recorded lateral stance/roll association,
not a proven cause. Static loads remain assumptions. Max40 unique19-knot checks,
10 SLSQP iterations, ONE91-knot check for each of four known failing geometries.
No simulation, runtime/control modification, optimizer learning, holdout or admission.
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
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'centered_orientation_refinement_227'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Budget(Exception):pass

def main():
 O.mkdir(exist_ok=False);mp=D/'cooked_model_50/model.json';ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';cp=D.parent/'contact_effort_repair_01/plane_calibration.json';m=BodyModel(mp);meta=read(mp);ine=read(ip);cal=read(cp)
 sources=[Path(__file__).resolve(),mp,ip,cp,D/'stance_diagnosis_219/results.json']+[Path(__file__).with_name(x) for x in ['low_seat_centered_coarse_223.py','low_seat_fixed_foot_plan_126.py','low_seat_balanced_force_v2_145.py','low_seat_touchdown_path_175.py']]+[D/'centered_path_check_221'/f'case_{c}.json' for c in [0,2,6,10]]
 bounds=[(-.28,-.1),(-np.deg2rad(12),np.deg2rad(12)),(np.deg2rad(5),np.deg2rad(38)),(-np.deg2rad(35),np.deg2rad(35))]
 rev=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,maximum_evaluations_per_geometry=40,maximum_iterations=10,dense_checks_per_geometry=1,variable_bounds=bounds,fixed_lateral_root_m=0.,research_foot_load_difference_BW=.1,research_ankle_roll_command_change_rad=.04,source_sha256={str(p.resolve()):sha(p) for p in sources},physics_ticks=0,optimizer_updates=0,holdout_used=False)
 (O/'revision.json').write_text(json.dumps(rev,indent=2));rows=[]
 for c in [0,2,6,10]:
  old=read(D/'centered_path_check_221'/f'case_{c}.json');x=old['inputs'];ep=np.array(old['endpoint_parameters']);history=[];cache={};roll_ids=[m.names.index(s+'_ankle_roll_joint') for s in ['left','right']]
  def parameters(z):return [float(z[0]),0.,float(z[1]),float(z[2]),float(z[3])]
  def build(z,fn):
   p=fn(m,meta,ine,cal,x['q'],x['gravity'],x['geometry'],x['reference'],x['command'],parameters(z));return reallocate(m,ine,p,x)
  def metrics(p):
   path=p['path'];loads=[e['load_checks'][0] for e in path];fractions=[{s:sum(w for w,ss in zip(l['corner_fractions'],l['corner_sides']) if ss==s) for s in ['left','right']} for l in loads];diff=max(abs(v['left']-v['right']) for v in fractions);roll=max(abs(np.array(l['command'])[roll_ids]-np.array(x['command'])[roll_ids]).max() for l in loads)
   cons=np.array([min(e['foot_cop_margin_m'] for e in path)-.005,min(e['patch_margin_m'] for e in path if e['pelvis_gap_m']<.02)-.015,min(e['floor_clearance_m'] for e in path)-.005,.003-max(e['foot_position_error_m'] for e in path),np.deg2rad(.5)-max(e['foot_orientation_error_rad'] for e in path),min(e['min_command_margin_rad'] for e in loads)-.03,1-max(e['maximum_effort_utilization'] for e in loads),min(v['margin_m'] for e in loads for v in e['individual_foot_cop'].values())-.005,.1-diff,.04-roll])
   return cons,float(diff),float(roll)
  def evaluate(z):
   key=tuple(float(v) for v in z)
   if key in cache:return cache[key]
   if len(cache)>=40:raise Budget()
   p=build(z,coarse);con,diff,roll=metrics(p);other=bool(all(e['other_model_passed'] and e['load_checks'][0]['passed'] for e in p['path']) and min(con)>=0)
   rec=dict(index=len(history),parameters=parameters(z),optimization_variables=list(key),model_passed=p['model_passed'],other_research_passed=other,constraints=con.tolist(),clearance_m=min(e['clearance_m'] for e in p['path']),maximum_assumed_foot_load_difference_BW=diff,maximum_ankle_roll_command_change_rad=roll,summary=p['summary']);history.append(rec);cache[key]=rec;(O/f'coarse_{c}_{rec["index"]:02d}.json').write_text(json.dumps(dict(case=c,inputs=x,endpoint_parameters=parameters(z),plan=p,record=rec),indent=2));print('COARSE',c,rec['index'],'clear_mm',1000*rec['clearance_m'],'other',other,'roll',roll,flush=True);return rec
  start=ep[[0,2,3,4]];result=None;budget=False
  try:result=minimize(lambda z:-100*evaluate(z)['clearance_m'],start,method='SLSQP',bounds=bounds,constraints=[dict(type='ineq',fun=lambda z:100*np.array(evaluate(z)['constraints']))],options=dict(maxiter=10,eps=1e-5,ftol=1e-8))
  except Budget:budget=True
  valid=[a for a in history if a['other_research_passed']];best=max(valid,key=lambda z:z['clearance_m']) if valid else max(history,key=lambda z:min(z['constraints']));p=build(best['optimization_variables'],dense);cons,diff,roll=metrics(p);research_pass=bool(p['model_passed'] and min(cons)>=0);schedule=dict(checked=False,physical_admission=False)
  if research_pass:
   engine=TouchdownPath(p,m);targets=[];nom=[]
   for tick in range(1,1501):
    engine.settle=dict(quiet=True);targets.append(engine.advance(True,.1).copy());nom.append(engine.nominal.copy())
    if engine.position>=engine.arc[-1]-1e-10:break
   schedule=dict(checked=True,synthetic_quiet_and_load=True,duration_s=tick*.02,ticks=tick,complete=bool(engine.position>=engine.arc[-1]-1e-10),time_gate_passed=bool(engine.position>=engine.arc[-1]-1e-10 and tick*.02<=13.5),physical_admission=False);np.savez_compressed(O/f'schedule_{c}.npz',targets=np.array(targets),nominal=np.array(nom),initial=np.array(x['command'],np.float32))
  summary=dict(case=c,evaluations=len(history),budget_reached=budget,optimizer_message=str(result.message) if result is not None else 'bounded evaluation cap',best_coarse=best,dense_summary=p['summary'],model_passed=p['model_passed'],research_pass=research_pass,dense_research_constraints=cons.tolist(),maximum_assumed_foot_load_difference_BW=diff,maximum_ankle_roll_command_change_rad=roll,schedule=schedule)
  (O/f'case_{c}.json').write_text(json.dumps(dict(case=c,inputs=x,endpoint_parameters=best['parameters'],plan=p,history=history,summary=summary,schedule=schedule),indent=2));rows.append(summary);print('DENSE',json.dumps(summary),flush=True)
 (O/'results.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,scope=__doc__,source_sha256=rev['source_sha256'],physics_ticks=0,optimizer_updates=0,physical_admission=False),indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()
