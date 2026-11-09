"""No-physics fallback identity and measured-current-view routing checks."""
import ast,json,hashlib,sys,types
from pathlib import Path
import numpy as np
import torch
from humanoid_sitting.shared_frame import BodyModel
from humanoid_sitting.supported_surface import observed_support
from diagnose_low_seat_visibility_38 import REAR,DOWN
from low_seat_dual_view_40 import fallback
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OUT=D/'dual_view_preparation_40'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
class Array:
 def __init__(self,x):self.torch=torch.as_tensor(x)
class Cam:
 def __init__(self,d,k):self.data=types.SimpleNamespace(output={'distance_to_image_plane':Array(d)},intrinsic_matrices=Array(k))
def main():
 audit=read(D/'dual_view_audit_39/independent_audit.json');assert audit['diagnostic_checks_passed']
 unavailable=audit['summaries']['target_rear_tracker_unavailable'];assert unavailable['frames']>0 and unavailable['down']['valid']==unavailable['frames']
 assert unavailable['down']['within_5mm_height_1deg_slope']==unavailable['frames']
 assert all(x['exact'] for x in audit['prefix_comparison_to_original36'].values())
 assert all(x['exact'] for values in audit['exact_starts'].values() for x in values.values())
 OUT.mkdir(exist_ok=False);model=BodyModel('runs/g1_kinematics_geometry_04/model.json');a=np.load(D/'dual_view_diagnostic_39/raw_dual_captures.npz');calls=[]
 sys.modules['run_supported_transition']=types.SimpleNamespace(base=types.SimpleNamespace(acquire_current_camera=lambda cam:calls.append('expose')))
 target=[0,1,2,3,6,7,10,11];records=[]
 for k in [0,20,35]:
  q=a['q'][k];g=a['gravity'][k];prior=a['geometry'][k];rear=[]
  for i in range(24):rear.append(observed_support(model,a['rear_K'][k,i],REAR,a['rear_depth'][k,i],q[i],g[i],prior[i])[0])
  duration=torch.full((24,),5.);duration[target]=14.
  env=types.SimpleNamespace(t=round(a['t'][k]*50),num_envs=24,staged_ready=True,repair_duration=duration,warming=torch.zeros(24,dtype=torch.long),trackers=[types.SimpleNamespace(value=x.copy()) for x in prior],model=model,raw=types.SimpleNamespace(sim=types.SimpleNamespace(get_physics_step_count=lambda:1234),scene={'rear_camera':Cam(a['rear_depth'][k],a['rear_K'][k]),'diagnostic_down_camera':Cam(a['down_depth'][k],a['down_K'][k])}),dual_records=[],dual_details=[])
  old=list(rear);before=len(calls);out=fallback(env,rear,q,g);needed=[i for i in target if old[i] is None]
  assert len(calls)-before==bool(needed)
  assert all(out[i] is old[i] for i in range(24) if i not in needed)
  for i in needed:
   independent=observed_support(model,a['down_K'][k,i],DOWN,a['down_depth'][k,i],q[i],g[i],prior[i])[0]
   assert independent is not None and np.array_equal(independent,out[i])
  env.repair_duration[:]=5.;before=len(calls);identity=fallback(env,list(old),q,g);assert len(calls)==before and all(x is y for x,y in zip(identity,old))
  records.append(dict(time_s=float(a['t'][k]),fallback_cases=needed,unchanged_rear_or_unselected_identity=True,independent_current_down_fit_exact=True,unselected_only_no_exposure=True))
 source=Path('scripts/run_supported_transition.py').read_text();tree=ast.parse(source);cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='TransitionEnv');method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_observe');original=ast.get_source_segment(source,method)
 needle='for i in range(self.num_envs):self.geometry[i]=self.trackers[i].update(q[i],gravity[i],gyro[i],measurement=measurements[i])';assert original.count(needle)==1
 insertion='measurements=_dual_view_fallback_40(self,measurements,q,gravity,force)\n        ';new=original.replace(needle,insertion+needle);assert new.replace(insertion,'')==original;compile(new,'<observe_equivalence>','exec')
 paths=[Path(__file__).resolve(),Path('scripts/low_seat_dual_view_40.py').resolve(),Path('scripts/run_low_seat_dual_view_40.py').resolve(),Path('scripts/run_supported_transition.py').resolve(),Path('sitting_task/src/humanoid_sitting/supported_surface.py').resolve(),D/'dual_view_audit_39/independent_audit.json',D/'dual_view_diagnostic_39/raw_dual_captures.npz']
 result=dict(passed=True,scope='Pure software/recorded-sensor routing checks, no physical repair admission. Down-view measurement-quality screen is5mm height/1deg slope on the recorded fallback frames, not a changed slip/contact threshold.',physics_ticks=0,optimizer_updates=0,source_transform_exact=True,records=records,source_input_hashes={str(p):sha(p) for p in paths})
 (OUT/'preparation.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
