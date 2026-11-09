"""Independent virtual-work replay and command-only filter screen; zero physics."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.sparse import load_npz
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from low_seat_joint_path_controller_17 import measured
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'contact_command_bound_audit_113'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);source=D/'contact_command_bound_112/results.json';data=read(source);pathsource=D/'stance_com_path_v2_98/results.json';pathdata=read(pathsource)['dense'];assert data['summary']['unchanged_residual_checks_passed'];
 for p,h in data['source_sha256'].items():assert sha(p)==h,p
 m=BodyModel(D/'cooked_model_50/model.json');ine=read(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json');names=m.names;mass=np.array(ine['mass']);W=mass.sum()*9.81;moving=[names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]+[names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']];kp=np.array([200,150,150,200,160,160]*2+[40]*4);effort=np.array([300,300,300,300,20,20]*2+[300]*4);limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'};fixed=[j for j in range(len(names)) if j not in moving];fingers=[i for i,n in enumerate(names) if any('_'+v+'_joint' in n for v in ['zero','one','two','three','four','five','six'])];assert len(fingers)==14;cmd0=np.array(pathdata['command_nodes'][0]['command']);q0=np.array(pathdata['path'][0]['q']);errors=[];eqs=[];commands=[];cm=[];use=[]
 src=D/'normal_landing3s_06/candidate';tr=np.load(src/'trace.npz');sensors=np.load(src/'sensor_inputs.npz')['sensors'];calpath=D.parent/'contact_effort_repair_01/plane_calibration.json';at=int(np.argmin(abs(tr['t']-1.)));case=2;startq=tr['q'][at,case].astype(float);assert np.array_equal(startq[:23].astype(np.float32),sensors[0,10,case,270:293]);n,d,_,_,_,_,_=measured(m,startq,sensors[0,10,case,339:342],tr['geometry'][at,case],[tr['estimated_root_x'][at,case],0,0],read(calpath))
 for k,(e,v,old) in enumerate(zip(pathdata['path'],data['nodes'],pathdata['command_nodes'])):
  q=np.array(e['q']);root=e['root'];T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);fk=m.fk(q);C=np.average([apply(T@fk[b],cc) for b,cc in zip(ine['body_names'],ine['local_com'])],axis=0,weights=mass);P=np.array(old['corner_world_points']);w=np.array(v['corner_fractions']);ss=old['corner_sides'];pv=apply(T,m.vertices['pelvis']);pd=pv@n+d;seat=pv[pd<=pd.min()+1e-7].mean(0);seat-=n*(seat@n+d);normal=old['assumed_seat_normal_fraction'];sv=normal/n[2];assert normal==0 or abs(pd.min())<1e-8;eq=max(abs(w.sum()+sv-1),max(abs((w[:,None]*P[:,:2]).sum(0)+sv*seat[:2]-C[:2])),abs(sum(ww for ww,s in zip(w,ss) if s=='right')-v['planned_right_load_fraction']));assert eq<1e-8 and min(w)>=-1e-10;eqs.append(eq)
  assert v['assumed_seat_normal_fraction']==old['assumed_seat_normal_fraction']
  if e['phase']=='swing' or k in [0,16]:assert abs(v['planned_right_load_fraction']-old['planned_right_load_fraction'])<1e-8
  if e['phase']=='swing':assert abs(v['planned_right_load_fraction'])<1e-8
  if e['phase'] in ['approach','load']:
   assert e['stance']==['left','right']
   for side in ['left','right']:
    assert np.array_equal(e['foot_targets'][side]['position'],pathdata['path'][-1]['foot_targets'][side]['position'])
    assert np.array_equal(e['foot_targets'][side]['rotation'],pathdata['path'][-1]['foot_targets'][side]['rotation'])
  if k==0:assert max(abs(w-np.array(old['corner_fractions'])))<1e-8
  local=[np.linalg.inv(T@fk[s+'_ankle_roll_link'])@np.r_[p,1.] for p,s in zip(P,ss)];tau=[]
  for j in moving:
   values=[]
   for sign in [-1,1]:
    qq=q.copy();qq[j]+=sign*1e-5;ff=m.fk(qq);V=sum(mm*9.81*apply(T@ff[b],cc)[2] for b,mm,cc in zip(ine['body_names'],mass,ine['local_com']));work=sum(W*ww*(T@ff[s+'_ankle_roll_link']@ll)[2] for ww,s,ll in zip(w,ss,local));values.append(V-work)
   tau.append((values[1]-values[0])/2e-5)
  tau=np.array(tau);err=max(abs(tau-v['torque_Nm']));errors.append(err);assert err<1e-6
  if k==0:tau0=tau.copy()
  expected=cmd0.copy();expected[moving]+=q[moving]-q0[moving]+(tau-tau0)/kp
  if k==0:expected=cmd0.copy()
  cmd=np.array(v['command']);assert max(abs(cmd-expected))<2e-8 and np.array_equal(cmd[fixed],cmd0[fixed]) and np.array_equal(cmd[fingers],cmd0[fingers]);commands.append(cmd);cm.append(min(min(cmd[j]-limits[names[j]][0],limits[names[j]][1]-cmd[j]) for j in moving));use.append(max(abs(tau)/effort))
 assert len(commands)==115
 commands=np.array(commands);arc=np.r_[0,np.cumsum(np.max(abs(np.diff(commands,axis=0)),axis=1))];assert abs(arc[-1]/.15-data['summary']['minimum_command_polyline_time_s'])<1e-10
 U=load_npz(D/'contact_command_bound_112/inequalities.npz');E=load_npz(D/'contact_command_bound_112/equalities.npz');a=np.load(D/'contact_command_bound_112/lp_witness.npz')
 x,c,ur,er,lo,hi,zi,ze,zl,zu=[a[k] for k in ['x','objective','inequality_rhs','equality_rhs','lower','upper','inequality_dual','equality_dual','lower_dual','upper_dual']]
 finite=np.isfinite(hi);primal=float(max(max(abs(E@x-er)),max(U@x-ur),max(lo-x),max(x[finite]-hi[finite]),0.));sign=float(max(max(zi),max(-zl),max(zu),0.));stationarity=float(max(abs(c-U.T@zi-E.T@ze-zl-zu)));dual=float(ur@zi+er@ze+lo@zl+hi[finite]@zu[finite]);gap=abs(float(c@x)-dual)
 assert primal<1e-8 and sign<1e-10 and stationarity<1e-8 and gap<1e-8 and np.all(zu[~finite]==0)
 assert abs(float(c@x)-arc[-1]/.15)<1e-8 and abs(dual-data['summary']['lp_objective_time_s'])<1e-8
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope='Independent all115-node finite-difference virtual work, equilibrium, initial/fixed commands, zero pre-lift/swing foot loading, preserved zero/assumed seat loading and model command constraints, plus saved GLOBAL LP primal/dual feasibility, signs, stationarity and duality gap. Certified numerical lower bound applies ONLY to visiting these115 fixed sampled poses in order under the unchanged static-command construction/contact force family. It is NOT a universal motion lower bound or dynamics/filtered execution certificate.',nodes=115,max_virtual_work_error_Nm=float(max(errors)),max_equilibrium_residual=float(max(eqs)),minimum_command_margin_rad=float(min(cm)),maximum_static_effort_utilization=float(max(use)),model_command_gate=bool(min(cm)>=.03 and max(use)<=1),specific_command_time_s=float(arc[-1]/.15),dual_time_lower_bound_s=dual,maximum_primal_residual=primal,dual_sign_violation=sign,maximum_stationarity=stationarity,duality_gap_s=gap,planning_allowance_s=13.5,command_time_gate=bool(arc[-1]/.15<=13.5),source_sha256={str(p.resolve()):sha(p) for p in [source,pathsource,Path(__file__),D/'contact_command_bound_112/inequalities.npz',D/'contact_command_bound_112/equalities.npz',D/'contact_command_bound_112/lp_witness.npz']},physics_ticks=0,optimizer_updates=0,holdout_used=False,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps({k:v for k,v in out.items() if k!='source_sha256'}),flush=True)
if __name__=='__main__':main()
