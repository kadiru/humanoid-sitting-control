"""Current link-velocity/joint-velocity consistency and exact-input PD diagnosis.
No new controls, sensor substitution, quiet-guard waiver or causal proof.
"""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'drive_diagnostic_246';O=D/'drive_diagnosis_248'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x):return dict(min=float(np.min(x)),max=float(np.max(x)),median=float(np.median(x)),mean=float(np.mean(x)))
def main():
 O.mkdir(exist_ok=False);audit=read(D/'drive_audit_247/independent_audit.json');s=np.load(P/'substep_states.npz');launch=np.load(P/'implicit_launches.npz');meta=read(P/'actuator_metadata.json');model=read(D/'cooked_model_50/model.json');names=meta['joint_names'];bn=meta['body_names'];jmap={x['name']:x for x in model['joints']}
 qd=s['qd'];poses=s['link_transforms'];vel=s['link_velocities'];ids=np.flatnonzero(s['t']>=11.-1e-9);assert len(ids)==401;rows=[];closure=np.zeros_like(qd,dtype=float);offaxis=np.zeros_like(qd,dtype=float)
 for j,name in enumerate(names):
  x=jmap[name];b0=bn.index(x['body0'][0]);b1=bn.index(x['body1'][0]);a=Rotation.from_quat(x['q0']).as_matrix()[:, 'XYZ'.index(x['axis'])]
  R=Rotation.from_quat(poses[:,:,b0,3:].reshape(-1,4)).as_matrix().reshape(len(s['t']),24,3,3);axis=np.einsum('...ij,j->...i',R,a);rel=vel[:,:,b1,3:].astype(float)-vel[:,:,b0,3:].astype(float);project=np.sum(rel*axis,axis=-1);closure[:,:,j]=project-qd[:,:,j];offaxis[:,:,j]=np.linalg.norm(rel-project[:,:,None]*axis,axis=-1)
  for c in range(24):
   delta=Rotation.from_quat(poses[ids[1:],c,b1,3:])*Rotation.from_quat(poses[ids[:-1],c,b1,3:]).inv();omega_fd=delta.as_rotvec()/.005
   rows.append(dict(case=c,joint=j,joint_name=name,parent=x['body0'][0],child=x['body1'][0],raw_qd_rad_s=stats(qd[ids,c,j]),projected_relative_link_angular_velocity_rad_s=stats(project[ids,c]),joint_link_velocity_residual_rad_s=stats(closure[ids,c,j]),off_axis_relative_angular_speed_rad_s=stats(offaxis[ids,c,j]),child_angular_speed_rad_s=stats(np.linalg.norm(vel[ids,c,b1,3:],axis=-1)),child_pose_difference_angular_speed_rad_s=stats(np.linalg.norm(omega_fd,axis=-1)),child_com_horizontal_speed_m_s=stats(np.linalg.norm(vel[ids,c,b1,:2],axis=-1))))
 li=np.flatnonzero(launch['t_before']>=11.-1e-9);effort=[]
 for c in range(24):
  for j,name in enumerate(names):
   pp=launch['kp'][li,c,j].astype(float)*(launch['target_q'][li,c,j].astype(float)-launch['cached_q'][li,c,j]);dd=launch['kd'][li,c,j].astype(float)*(launch['target_qd'][li,c,j].astype(float)-launch['cached_qd'][li,c,j]);ff=launch['target_effort'][li,c,j]
   effort.append(dict(case=c,joint=j,joint_name=name,position_component_Nm=stats(pp),velocity_component_Nm=stats(dd),feedforward_Nm=stats(ff),computed_telemetry_Nm=stats(launch['computed_effort'][li,c,j]),applied_telemetry_Nm=stats(launch['applied_effort'][li,c,j])))
 settings=read(P/'live_stage_settings.json');out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,window_s=[11,13],samples=401,rows=rows,implicit_components=effort,global_joint_link_velocity_residual_max_rad_s=float(abs(closure).max()),global_off_axis_relative_angular_speed_max_rad_s=float(offaxis.max()),exact_prefix_replay=audit['exact_prefix_replay'],diagnostic_checks_passed=audit['diagnostic_checks_passed'],live_authored_settings=settings,physics_ticks=0,optimizer_updates=0,physical_admission=False,interpretation='Link angular velocities and joint velocities are simultaneous backend solver outputs. Their projection onto the recorded joint axis tests kinematic consistency, not equality to pose increments or proof of continuous motion. Exact-input PD terms reconstruct approximate telemetry only, not solver applied motor effort. USD settings describe instantiated attributes; effective internal island iteration counts remain unqueried. No controller modification or quiet/physical threshold change follows automatically.',source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),P/'substep_states.npz',P/'implicit_launches.npz',P/'actuator_metadata.json',P/'live_stage_settings.json',D/'cooked_model_50/model.json',D/'drive_audit_247/independent_audit.json']})
 (O/'results.json').write_text(json.dumps(out,indent=2));np.savez_compressed(O/'velocity_closure.npz',t=s['t'],joint_link_residual=closure,off_axis_speed=offaxis)
 fig,ax=plt.subplots(2,2,figsize=(12,7))
 for col,c in enumerate([0,1]):
  for j in [19,20]:
   x=jmap[names[j]];b1=bn.index(x['body1'][0]);ax[0,col].plot(s['t'][ids],qd[ids,c,j],label=names[j].split('_')[0]+' raw qd');ax[1,col].plot(s['t'][ids],np.linalg.norm(vel[ids,c,b1,3:],axis=1),label=names[j].split('_')[0]+' foot angular speed')
  ax[0,col].axhline(.05,color='black',linestyle='--',label='unchanged quiet-speed limit');ax[0,col].set_title(f'Centered case{c}');ax[0,col].set_ylabel('Joint speed(rad/s)');ax[1,col].set_ylabel('Foot angular speed(rad/s)')
  for a in ax[:,col]:a.legend(fontsize=8);a.grid(alpha=.2);a.set_xlabel('Time(s)')
 fig.suptitle('Private simultaneous solver velocities — compare pose differences separately\nDiagnostic only; no physical repair or guard waiver');fig.tight_layout();fig.savefig(O/'link_velocity.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,4));rr=[r for r in effort if r['case'] in [0,1] and r['joint'] in [19,20]];x=np.arange(4);ax.bar(x-.2,[r['position_component_Nm']['median'] for r in rr],.2,label='position term');ax.bar(x,[r['velocity_component_Nm']['median'] for r in rr],.2,label='velocity term');ax.bar(x+.2,[r['applied_telemetry_Nm']['median'] for r in rr],.2,label='approximate PD telemetry');ax.set_xticks(x,[f"case{r['case']} {r['joint_name'].split('_')[0]}" for r in rr]);ax.set_ylabel('Nm');ax.legend();ax.grid(axis='y',alpha=.2);ax.set_title('Precisely captured implicit-kernel inputs,11–13s\nNot direct solver or hardware motor effort');fig.tight_layout();fig.savefig(O/'exact_input_effort.png',dpi=160);plt.close(fig)
 (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(dict(exact_prefix=out['exact_prefix_replay'],closure_max=out['global_joint_link_velocity_residual_max_rad_s'],selected=[r for r in rows if r['case'] in [0,1] and r['joint'] in [19,20]])),flush=True)
if __name__=='__main__':main()
