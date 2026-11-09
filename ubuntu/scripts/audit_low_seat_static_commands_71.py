"""Independent static-work, equilibrium, actuator-bound and command audit."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'static_command_audit_71'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 O.mkdir(exist_ok=False);p=D/'contact_command_screen_68/results.json';res=read(p)
 for f,h in res['source_sha256'].items():assert sha(f)==h,f
 m=BodyModel(D/'cooked_model_50/model.json');ine=read(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json');W=sum(ine['mass'])*9.81;names=m.names;moving=[names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]+[names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']];kp=np.array([200,150,150,200,160,160]*2+[40]*4);effort=np.array([300,300,300,300,20,20]*2+[300]*4);fixed=[j for j in range(len(names)) if j not in moving];fingers=[i for i,n in enumerate(names) if any('_'+s+'_joint' in n for s in ['zero','one','two','three','four','five','six'])];assert len(fingers)==14
 limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'};tr=np.load(D/'normal_landing3s_06/candidate/trace.npz');at=int(np.argmin(abs(tr['t']-1)));q0=tr['q'][at,2].astype(float);command0=tr['command'][at,2].astype(float);out=[]
 unitree=Path('IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py').read_text();full=Path('sitting_task/src/humanoid_sitting/tasks/full_transition.py').read_text();assert 'joint_effort_limit=20' in unitree and 'joint_effort_limit=300' in unitree and "self.scene.robot.actuators['feet'].stiffness=160." in full
 for row in res['rows']:
  assert row['moving_joint_names']==[names[j] for j in moving];assert np.array_equal(row['kp_Nm_rad'],kp) and np.array_equal(row['configured_effort_limits_Nm'],effort)
  src=read(row['source']);assert sha(row['source'])==row['source_sha256'];path=src['rows'][0]['path'] if 'rows' in src else src['dense']['path'];assert len(path)==len(row['nodes'])==105;errors=[];equilib=[];commands=[];fail=[];bounds=[];tau0=None;exceed=[]
  for e,a in zip(path,row['nodes']):
   q=np.array(e['q']);root=e['root'];T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);fk=m.fk(q);C=np.average([apply(T@fk[b],cc) for b,cc in zip(ine['body_names'],ine['local_com'])],axis=0,weights=ine['mass']);P=np.array(a['corner_world_points']);weights=np.array(a['corner_fractions']);ss=a['corner_sides'];assert len(weights)==len(P)==len(ss) and min(weights)>=-1e-10;eq=max(abs(weights.sum()-1),float(max(abs((weights[:,None]*P[:,:2]).sum(0)-C[:2]))));assert eq<1e-8;equilib.append(eq);assert abs(sum(w for w,s in zip(weights,ss) if s=='right')-a['planned_right_load_fraction'])<1e-8
   if e['phase']=='swing':assert all(abs(w)<1e-8 for w,s in zip(weights,ss) if s=='right')
   local=[np.linalg.inv(T@fk[s+'_ankle_roll_link'])@np.r_[p,1.] for p,s in zip(P,ss)];tau=[]
   for j in moving:
    V=[]
    for sign in [-1,1]:
     qq=q.copy();qq[j]+=sign*1e-5;ff=m.fk(qq);potential=sum(mm*9.81*apply(T@ff[b],cc)[2] for b,mm,cc in zip(ine['body_names'],ine['mass'],ine['local_com']));work=sum(W*w*(T@ff[s+'_ankle_roll_link']@v)[2] for w,s,v in zip(weights,ss,local));V.append(potential-work)
    tau.append((V[1]-V[0])/2e-5)
   tau=np.array(tau);errors.append(float(max(abs(tau-a['planned_static_torque_Nm']))));assert errors[-1]<1e-6
   if tau0 is None:tau0=tau.copy()
   expected=command0.copy();expected[moving]+=q[moving]-q0[moving]+(tau-tau0)/kp;cmd=np.array(a['command']);assert max(abs(cmd-expected))<2e-8;assert np.array_equal(cmd[fixed],command0[fixed]) and np.array_equal(cmd[fingers],command0[fingers]);commands.append(cmd)
   cm=float(min(min(cmd[j]-limits[names[j]][0],limits[names[j]][1]-cmd[j]) for j in moving));assert abs(cm-a['joint_command_margin_rad'])<1e-12;bad=[names[moving[i]] for i in np.flatnonzero(abs(tau)>effort+1e-7)];assert bad==a['torque_limit_exceed_names'];exceed+=bad;bounds.append(cm);passed=bool(a['allocation']['passed'] and cm>=.03 and not bad);assert a['static_model_passed']==passed
   if not passed:fail.append(a['knot'])
  assert fail==row['failed_static_command_nodes'];Q=np.array(commands);assert np.array_equal(Q[0],command0);arc=float(np.max(abs(np.diff(Q,axis=0)),axis=1).sum());assert abs(arc-row['command_arc_rad'])<1e-12;assert abs(arc/.15-row['minimum_time_for_this_command_polyline_at015_s'])<1e-12
  out.append(dict(source=row['source'],nodes=105,independent_virtual_work_max_error_Nm=max(errors),max_equilibrium_residual=max(equilib),failed_command_nodes=fail,minimum_command_joint_margin_rad=min(bounds),torque_limit_exceed_names=sorted(set(exceed)),command_polyline_minimum_duration_s=arc/.15,static_command_gate_passed=not fail,all14_fingers_exact=True,initial_command_exact=True))
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope='Independent finite-difference virtual work, nonnegative vertical-force equilibrium, planned unloaded swing, all16 moving commands, configured actuator effort/joint bounds, exact recorded initial command/all14 fixed fingers and specific-command-polyline timing on210saved nodes. Both failed command gates remain failed. This is a static audit, not sensed contact, dynamic feasibility, friction capacity or physical admission.',rows=out,source_sha256={str(p):sha(p),str(Path(__file__).resolve()):sha(__file__)},physics_ticks=0,optimizer_updates=0,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(result),flush=True)
if __name__=='__main__':main()
