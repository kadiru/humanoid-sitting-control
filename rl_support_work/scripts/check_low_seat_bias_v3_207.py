"""Bounded integral arithmetic, failure gates and recorded-input counterfactual.
No environment is created; recorded inputs are never installed into runtime.
"""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel
from low_seat_bias_feedback_v2_207 import BiasFeedback
from low_seat_balance_feedback_160 import BalanceFeedback
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'bias_feedback_checks_v3_207'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);m=BodyModel(D/'cooked_model_50/model.json')
 plans={r['case']:r['plan'] for r in read(D/'touchdown_probe_178/candidate/fixed_plans_132.json') if r['plan']['model_passed']}
 records=read(D/'touchdown_probe_178/candidate/balance_feedback_161.json')
 from low_seat_touchdown_path_175 import TouchdownPath
 engines={i:TouchdownPath(p,m) for i,p in plans.items()}
 fs={i:BiasFeedback(m,p) for i,p in plans.items()};old={i:BalanceFeedback(m,p) for i,p in plans.items()}
 comparisons=[];counterfactual=[]
 for r in records:
  i=r['case'];f=fs[i];hover=abs(r['position']-engines[i].hover)<1e-10
  args=[np.array(r[k],np.float32) for k in ['q','gravity','gyro']]+[r['position'],np.array(r['nominal_target'],np.float32),np.array(r['foot_wrench']),np.array(r['conditions']),r['weight']]
  before=f.integral;tar,a=f.update(*args,geometry=np.array(r['geometry'],np.float32),at_hover=hover)
  baseline,b=old[i].update(*args)
  assert b==r['feedback'] and np.array_equal(baseline,np.array(r['target'],np.float32))
  if not hover and before==0:assert np.array_equal(tar,baseline)
  check_bounds(f,tar,np.array(r['nominal_target'],np.float32),a)
  err=float(np.arctan2(args[1][0],-args[1][2]))-a['target_pitch_rad']
  expected=float(np.clip(before+.3*err*.02,-.03,.03)) if a['integral_update_accepted'] else before
  assert f.integral==expected
  counterfactual.append(dict(case=i,tick=r['tick'],at_hover=hover,integral=f.integral,antiwindup=a['integral_antiwindup'],old_target_max_difference_rad=float(abs(tar-baseline).max())))
  comparisons.append(not hover and before==0)
 modes=[];fixture=next(r for r in records if r['case']==0 and r['tick']==600)
 for mode in ['negative_bias','positive_bias','large_error','stale','unloaded','fast_gyro','authority_saturation']:
  f=BiasFeedback(m,plans[0]);pos=engines[0].hover;nom=np.array(fixture['nominal_target'],np.float32);q=np.array(fixture['q'],np.float32);geo=np.array(fixture['geometry'],np.float32);geo[11:]=[1,0]
  goal=float(np.interp(pos,f.arc,f.pitch));err=np.deg2rad(1.35)*(-1 if mode!='positive_bias' else 1)
  if mode=='large_error':err=np.deg2rad(4.)
  g=-Rotation.from_euler('y',goal+err).as_matrix().T[:,2];gyro=np.zeros(3);wrench=np.array([[0.,50.,0.],[0.,50.,0.]])
  if mode=='stale':geo[12]=.6
  if mode=='unloaded':wrench[:,1]=0
  if mode=='fast_gyro':gyro[1]=.1
  if mode=='authority_saturation':f.tau[:]=18.
  accepted=0;windup=0
  for tick in range(600):
   tar,a=f.update(q,g,gyro,pos,nom,wrench,np.array([20.,20.]),100.,geometry=geo,at_hover=True)
   check_bounds(f,tar,nom,a);accepted+=a['integral_update_accepted'];windup+=a['integral_antiwindup']
  if mode in ['large_error','stale','unloaded','fast_gyro']:assert f.integral==0 and accepted==0
  if mode in ['negative_bias','positive_bias']:assert abs(f.integral)==.03
  if mode=='authority_saturation':assert windup>0 and abs(f.integral)<.03
  saved=f.integral;geo[12]=.6
  for _ in range(100):
   tar,a=f.update(q,g,gyro,pos,nom,wrench,np.array([20.,20.]),100.,geometry=geo,at_hover=True)
   assert f.integral==saved and a['integral_applied_rad']==0 and not a['integral_update_accepted']
  modes.append(dict(mode=mode,integral_rad=saved,accepted_ticks=accepted,antiwindup_ticks=windup,stale_state_frozen=True))
 sources=[Path(__file__).resolve(),Path('scripts/low_seat_bias_feedback_v2_207.py').resolve(),Path('scripts/low_seat_balance_feedback_160.py').resolve(),D/'touchdown_probe_178/candidate/balance_feedback_161.json',D/'touchdown_probe_178/candidate/fixed_plans_132.json']
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,recorded_input_rows=len(records),zero_integral_outside_hover_exact_rows=sum(comparisons),counterfactual_not_physics=True,modes=modes,physics_ticks=0,optimizer_updates=0,physical_admission=False,source_sha256={str(p):sha(p) for p in sources})
 (O/'counterfactual.json').write_text(json.dumps(counterfactual));(O/'independent_checks.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2));print(json.dumps(out),flush=True)
def check_bounds(f,tar,nom,a):
 assert np.isfinite(tar).all() and abs(f.integral)<=.03
 assert max(abs(np.array(a['offset_after'])))<=.06+1e-7
 assert max(abs(np.array(a['offset_after'])-np.array(a['offset_before'])))<=.00075+1e-7
 ids=[i for i in range(37) if i not in f.ids];assert np.array_equal(tar[ids],nom[ids])
 assert np.all(tar[f.ids]>=f.lo-3e-7) and np.all(tar[f.ids]<=f.hi+3e-7)
if __name__=='__main__':main()
