"""Bounded OFFLINE reduction of lateral stance demand, motivated by recorded219.
Keep old complete fixed-foot path family, roll/yaw, arms and force objective. Vary
ONLY root-x/pitch within existing bounds and root-y BETWEEN center and the old
50mm endpoint. Minimize squared lateral shift subject to ALL existing path gates
and a separate 3.2mm model clearance reserve. No physical threshold is changed.
Two geometries, max40 unique19-knot candidates/10iterations plus one91-knot dense
check each. No physics, runtime installation, training, holdout or admission.
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
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'lateral_demand_refinement_225'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Budget(Exception):pass

def main():
 O.mkdir(exist_ok=False);mp=D/'cooked_model_50/model.json';ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';cp=D.parent/'contact_effort_repair_01/plane_calibration.json';fp=D/'bias_probe_210/candidate/fixed_plans_132.json';m=BodyModel(mp);meta=read(mp);ine=read(ip);cal=read(cp);plans={r['case']:r for r in read(fp)}
 sources=[Path(__file__).resolve(),mp,ip,cp,fp,D/'stance_diagnosis_219/results.json']+[Path(__file__).with_name(x) for x in ['low_seat_centered_coarse_223.py','low_seat_fixed_foot_plan_126.py','low_seat_balanced_force_v2_145.py','low_seat_touchdown_path_175.py']]
 rev=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,maximum_evaluations_per_geometry=40,maximum_iterations=10,dense_checks_per_geometry=1,root_x_bounds_m=[-.28,-.1],pitch_bounds_deg=[5,38],lateral_bounds='between zero and each saved50mm endpoint',model_clearance_reserve_m=.0032,unchanged_model_clearance_gate_m=.003,source_sha256={str(p.resolve()):sha(p) for p in sources},physics_ticks=0,optimizer_updates=0,holdout_used=False)
 (O/'revision.json').write_text(json.dumps(rev,indent=2));rows=[]
 for c in [0,6]:
  old=plans[c];x=old['inputs'];ep=np.array(old['plan']['path'][-1]['parameters'][:5]);history=[];cache={};yb=sorted([0.,ep[1]])
  def parameters(z):return [float(z[0]),float(z[2]),float(ep[2]),float(z[1]),float(ep[4])]
  def build(z,fn):
   p=fn(m,meta,ine,cal,x['q'],x['gravity'],x['geometry'],x['reference'],x['command'],parameters(z));return reallocate(m,ine,p,x)
  def evaluate(z):
   key=tuple(float(v) for v in z)
   if key in cache:return cache[key]
   if len(cache)>=40:raise Budget()
   p=build(z,coarse);path=p['path'];loads=[e['load_checks'][0] for e in path]
   constraints=np.array([min(e['clearance_m'] for e in path)-.0032,min(e['foot_cop_margin_m'] for e in path)-.005,min(e['patch_margin_m'] for e in path if e['pelvis_gap_m']<.02)-.015,min(e['floor_clearance_m'] for e in path)-.005,.003-max(e['foot_position_error_m'] for e in path),np.deg2rad(.5)-max(e['foot_orientation_error_rad'] for e in path),min(e['min_command_margin_rad'] for e in loads)-.03,1-max(e['maximum_effort_utilization'] for e in loads),min(v['margin_m'] for e in loads for v in e['individual_foot_cop'].values())-.005])
   fractions=[{s:sum(w for w,ss in zip(l['corner_fractions'],l['corner_sides']) if ss==s) for s in ['left','right']} for l in loads];roll_ids=[m.names.index(s+'_ankle_roll_joint') for s in ['left','right']]
   rec=dict(index=len(history),parameters=parameters(z),optimization_variables=list(key),model_passed=p['model_passed'],research_constraints_passed=bool(min(constraints)>=0),constraints=constraints.tolist(),lateral_demand_m=abs(z[2]),maximum_assumed_foot_load_difference_BW=max(abs(v['left']-v['right']) for v in fractions),maximum_ankle_roll_command_change_rad=max(abs(np.array(l['command'])[roll_ids]-np.array(x['command'])[roll_ids]).max() for l in loads),summary=p['summary'])
   history.append(rec);cache[key]=rec;(O/f'coarse_{c}_{rec["index"]:02d}.json').write_text(json.dumps(dict(case=c,inputs=x,endpoint_parameters=parameters(z),plan=p,record=rec),indent=2));print('COARSE',c,rec['index'],'y',z[2],'clear_mm',p['summary']['minimum_housing_clearance_mm'],'pass',rec['research_constraints_passed'],flush=True);return rec
  zero=np.array([ep[0],ep[3],0.]);first=evaluate(zero);result=None;budget=False
  if not first['research_constraints_passed']:
   try:result=minimize(lambda z:(z[2]/.05)**2,np.array([ep[0],ep[3],ep[1]]),method='SLSQP',bounds=[(-.28,-.1),(np.deg2rad(5),np.deg2rad(38)),yb],constraints=[dict(type='ineq',fun=lambda z:100*np.array(evaluate(z)['constraints']))],options=dict(maxiter=10,eps=1e-5,ftol=1e-8))
   except Budget:budget=True
  valid=[z for z in history if z['research_constraints_passed']];best=min(valid,key=lambda z:z['lateral_demand_m']) if valid else max(history,key=lambda z:min(z['constraints']));p=build(best['optimization_variables'],dense);schedule=dict(checked=False,physical_admission=False)
  research_pass=bool(p['model_passed'] and p['summary']['minimum_housing_clearance_mm']>=3.2)
  if research_pass:
   engine=TouchdownPath(p,m);targets=[];nom=[]
   for tick in range(1,1501):
    engine.settle=dict(quiet=True);targets.append(engine.advance(True,.1).copy());nom.append(engine.nominal.copy())
    if engine.position>=engine.arc[-1]-1e-10:break
   schedule=dict(checked=True,synthetic_quiet_and_load=True,duration_s=tick*.02,ticks=tick,complete=bool(engine.position>=engine.arc[-1]-1e-10),time_gate_passed=bool(engine.position>=engine.arc[-1]-1e-10 and tick*.02<=13.5),physical_admission=False);np.savez_compressed(O/f'schedule_{c}.npz',targets=np.array(targets),nominal=np.array(nom),initial=np.array(x['command'],np.float32))
  summary=dict(case=c,evaluations=len(history),budget_reached=budget,optimizer_message=str(result.message) if result is not None else 'zero lateral is feasible or bounded cap',original_endpoint_parameters=ep.tolist(),best_coarse=best,dense_summary=p['summary'],model_passed=p['model_passed'],research_pass=research_pass,schedule=schedule)
  (O/f'case_{c}.json').write_text(json.dumps(dict(case=c,inputs=x,endpoint_parameters=best['parameters'],plan=p,history=history,summary=summary,schedule=schedule),indent=2));rows.append(summary);print('DENSE',json.dumps(summary),flush=True)
 (O/'results.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,scope=__doc__,source_sha256=rev['source_sha256'],physics_ticks=0,optimizer_updates=0,physical_admission=False),indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()
