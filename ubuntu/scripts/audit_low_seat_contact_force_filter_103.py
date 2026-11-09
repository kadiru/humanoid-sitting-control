"""Independent virtual-work replay and command-only filter screen; zero physics."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from low_seat_joint_path_controller_17 import measured
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'contact_force_filter_audit_103'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);source=D/'contact_force_schedule_102/results.json';data=read(source);pathsource=D/'stance_com_path_v2_98/results.json';pathdata=read(pathsource)['dense'];assert data['summary']['unchanged_residual_checks_passed'];
 for p,h in data['source_sha256'].items():assert sha(p)==h,p
 m=BodyModel(D/'cooked_model_50/model.json');ine=read(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json');names=m.names;mass=np.array(ine['mass']);W=mass.sum()*9.81;moving=[names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]+[names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']];kp=np.array([200,150,150,200,160,160]*2+[40]*4);effort=np.array([300,300,300,300,20,20]*2+[300]*4);limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'};fixed=[j for j in range(len(names)) if j not in moving];fingers=[i for i,n in enumerate(names) if any('_'+v+'_joint' in n for v in ['zero','one','two','three','four','five','six'])];assert len(fingers)==14;cmd0=np.array(pathdata['command_nodes'][0]['command']);q0=np.array(pathdata['path'][0]['q']);errors=[];eqs=[];commands=[];cm=[];use=[]
 src=D/'normal_landing3s_06/candidate';tr=np.load(src/'trace.npz');sensors=np.load(src/'sensor_inputs.npz')['sensors'];calpath=D.parent/'contact_effort_repair_01/plane_calibration.json';at=int(np.argmin(abs(tr['t']-1.)));case=2;startq=tr['q'][at,case].astype(float);assert np.array_equal(startq[:23].astype(np.float32),sensors[0,10,case,270:293]);n,d,_,_,_,_,_=measured(m,startq,sensors[0,10,case,339:342],tr['geometry'][at,case],[tr['estimated_root_x'][at,case],0,0],read(calpath))
 for k,(e,v,old) in enumerate(zip(pathdata['path'],data['nodes'],pathdata['command_nodes'])):
  q=np.array(e['q']);root=e['root'];T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);fk=m.fk(q);C=np.average([apply(T@fk[b],cc) for b,cc in zip(ine['body_names'],ine['local_com'])],axis=0,weights=mass);P=np.array(old['corner_world_points']);w=np.array(v['corner_fractions']);ss=old['corner_sides'];pv=apply(T,m.vertices['pelvis']);pd=pv@n+d;seat=pv[pd<=pd.min()+1e-7].mean(0);seat-=n*(seat@n+d);normal=old['assumed_seat_normal_fraction'];sv=normal/n[2];assert normal==0 or abs(pd.min())<1e-8;eq=max(abs(w.sum()+sv-1),max(abs((w[:,None]*P[:,:2]).sum(0)+sv*seat[:2]-C[:2])),abs(sum(ww for ww,s in zip(w,ss) if s=='right')-old['planned_right_load_fraction']));assert eq<1e-8 and min(w)>=-1e-10;eqs.append(eq)
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
 # Exact source order: desired clamp at hard limits+.03/-.03, then alpha .1,
 # then per20ms increment cap .003. This CPUfloat32 arithmetic is not a GPU replay.
 source_runtime=Path('scripts/run_seat_load_ppo.py');s=source_runtime.read_text();assert "clamp(min=limits[...,0]+.03,max=limits[...,1]-.03)" in s and "(.1*(desired-self.previous[:,self.ids])).clamp(min=-cap,max=cap)" in s
 duration=arc[-1]/.15;t=np.arange(0,np.ceil(duration/.02)*.02+1e-8,.02);desired=np.stack([np.interp(np.minimum(t*.15,arc[-1]),arc,commands[:,j]) for j in range(len(names))],axis=1).astype(np.float32);prev=desired[0].copy();lo=np.array([limits[names[j]][0]+.03 for j in moving],np.float32);hi=np.array([limits[names[j]][1]-.03 for j in moving],np.float32);raw=[];applied=[prev.copy()];clamp_events=[]
 for k in range(1,len(t)):
  target=prev.copy();target[moving]=prev[moving]+(desired[k,moving]-prev[moving])/np.float32(.1);clamped=np.clip(target[moving],lo,hi);bad=(target[moving]<lo)|(target[moving]>hi)
  if bad.any():clamp_events.append(dict(t=float(t[k]),joint_names=[names[moving[i]] for i in np.flatnonzero(bad)],maximum_target_excess_rad=float(max(np.maximum(lo-target[moving],target[moving]-hi)))))
  prev[moving]=prev[moving]+np.clip(np.float32(.1)*(clamped-prev[moving]),np.float32(-.003),np.float32(.003));raw.append(target.copy());applied.append(prev.copy())
 applied=np.array(applied);assert np.array_equal(applied[:,fixed],np.broadcast_to(cmd0.astype(np.float32)[fixed],applied[:,fixed].shape));maxerr=float(max(abs(applied-desired).ravel()));f=dict(scope='CPUfloat32 command arithmetic only, no physical integration or GPU exactness claim. Diagnostic inverse-filter targets expose clamping; no runtime target is installed.',sample_period_s=.02,alpha=.1,increment_cap_rad=.003,schedule_duration_s=float(t[-1]),inverse_target_clipped_ticks=len(clamp_events),events=clamp_events,max_command_path_difference_rad=maxerr,final_command_difference_rad=float(max(abs(applied[-1]-desired[-1]))),max_applied_increment_rad=float(max(abs(np.diff(applied,axis=0)).ravel())),unclipped_inverse_filter_gate=not clamp_events,physical_admission=False)
 np.savez_compressed(O/'command_only_replay.npz',t=t,model_command=desired,inverse_target=np.array(raw),applied=applied)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope='Independent finite-difference virtual work, force balance, fixed/finger commands and model timing for every115 saved node. The filter feasibility result is separate and may fail; an audit pass never admits physics.',nodes=115,max_virtual_work_error_Nm=float(max(errors)),max_equilibrium_residual=float(max(eqs)),minimum_command_margin_rad=float(min(cm)),maximum_static_effort_utilization=float(max(use)),model_command_gate=bool(min(cm)>=.03 and max(use)<=1),specific_command_time_s=float(arc[-1]/.15),command_time_gate=bool(arc[-1]/.15<=13.5),command_only_filter=f,source_sha256={str(source):sha(source),str(pathsource):sha(pathsource),str(source_runtime.resolve()):sha(source_runtime),str(Path(__file__).resolve()):sha(__file__)},physics_ticks=0,optimizer_updates=0,holdout_used=False,physical_admission=False);(O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps({k:v for k,v in out.items() if k not in ['source_sha256','command_only_filter']}));print(json.dumps({k:v for k,v in f.items() if k!='events'}))
if __name__=='__main__':main()
