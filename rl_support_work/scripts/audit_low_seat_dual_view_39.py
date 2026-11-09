"""Independent audit of the13s recording-only second-camera diagnostic.
No new physics; unchanged plane fitter selects without truth. Truth used only
for errors after selection. This is not a20s physical repair evaluation.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel
from humanoid_sitting.supported_surface import observed_support
from low_seat_joint_path_controller_17 import measured
from posture_episode_ppo import PosturePPO
from diagnose_low_seat_visibility_38 import REAR,DOWN
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'dual_view_diagnostic_39';OUT=D/'dual_view_audit_39'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def diff(a,b):
 same=(a==b)|(np.isnan(a)&np.isnan(b));finite=np.isfinite(a)&np.isfinite(b)
 return dict(exact=bool(same.all()),changed=int((~same).sum()),max_finite_difference=float(abs(a[finite]-b[finite]).max()) if finite.any() else None)
def main():
 OUT.mkdir(exist_ok=False);torch.set_num_threads(2)
 a=np.load(P/'raw_dual_captures.npz');account=read(P/'diagnostic_accounting.json');proto=read(P/'diagnostic_protocol.json');step=read(P/'step_status.json')
 assert account['active_ticks']==15600 and account['physics_substeps']==2600 and account['initial_settling_excluded']==600
 assert account['complete20s_episodes']==0 and account['optimizer_updates']==0 and not account['fresh_holdout'] and not account['down_depth_control_use']
 assert step['completed_control_steps']==130 and a['t'].shape==(36,) and np.allclose(a['t'],np.arange(6,13.01,.2),atol=1e-8)
 assert all(sha(p)==h for p,h in proto['sources'].items())
 original=D/'staged_approach_36/candidate';r=read(original/'results.json');geom=read(P/'geometry.json');origins=np.array(r['origins']);j=r['body_names'].index('pelvis');body=a['body'][:,:,j]
 R=Rotation.from_quat(body[...,3:].reshape(-1,4)).as_matrix().reshape(36,24,3,3);poses={}
 for key,mount in [('rear',REAR),('down',DOWN)]:
  pos=body[...,:3]+np.einsum('...ij,j->...i',R,mount[:3,3]);RC=Rotation.from_quat(a[key+'_orientation'].reshape(-1,4)).as_matrix().reshape(36,24,3,3)
  pe=float(abs(pos-a[key+'_position']).max());re=float(abs(RC-R@mount[:3,:3]@np.diag([1,-1,-1])).max())
  assert pe<1e-6 and re<1e-6
  poses[key]=dict(passed=True,position_max_error_m=pe,rotation_matrix_max_error=re,samples=864,quaternion_order='XYZW',raw_camera_frame='OpenGL; optical ray frame ROS')
 #Compare all recorded short-prefix arrays without forcing exactness.
 tr=np.load(P/'trace.npz');old=np.load(original/'trace.npz');assert len(tr['t'])==130
 replay={k:diff(tr[k],old[k][:130]) for k in ['t','body','q','command','geometry','reference','torque']}
 starts={}
 for name in ['startup.npz','traction_startup.npz','bridge_startup.npz']:
  x=np.load(P/name);y=np.load(original/name);starts[name]={k:diff(x[k],y[k]) for k in x.files if k in y.files}
 sensor=np.load(P/'actual_sensor_frames.npz')['sensors'];assert sensor.shape==(130,24,414) and np.isfinite(sensor).all()
 act=read(P/'parent_actions.json');assert len(act)==1 and act[0]['step']==110
 obs=np.asarray(act[0]['observation'],np.float32);assert np.array_equal(obs,sensor[110])
 cmd=read(D/'staged_approach_36/candidate.command.json');checkpoint=cmd[cmd.index('--checkpoint')+1];assert sha(checkpoint)==proto['parent_sha256']
 parent=PosturePPO(np.zeros(414),np.ones(414),0);parent.load(checkpoint)
 with torch.no_grad():pred=parent.choose(torch.from_numpy(obs),False)[0].numpy()
 actual=np.asarray(act[0]['action'],np.float32);assert np.allclose(pred,actual,atol=1e-5,rtol=1e-5)
 parentcheck=dict(passed=True,actual414_identity=True,checkpoint_sha256=sha(checkpoint),cpu_action_max_error=float(abs(pred-actual).max()),action_atol=1e-5,action_rtol=1e-5,decisions=1,new_actor_updates=0)
 #Independent fits use no previous plane. No claim of exact tracker replay.
 model=BodyModel('runs/g1_kinematics_geometry_04/model.json');cal=read(D.parent/'contact_effort_repair_01/plane_calibration.json');rows=[];fits=np.full((36,24,2,13),np.nan,np.float32)
 for k,t in enumerate(a['t']):
  for i in range(24):
   truth=geom[r['episodes'][i]['geometry_index']];root=body[k,i,:3]-origins[i];normal=R[k,i].T@truth['normal'];offset=float((root-truth['top_center'])@truth['normal'])
   row=dict(time_s=float(t),case=i,height_cm=truth['height']*100,slope_deg=truth['pitch_deg'],recorded_rear_tracker_valid=bool(a['geometry'][k,i,11]),recorded_rear_tracker_age_s=float(a['geometry'][k,i,12]),views={})
   for vi,(key,mount) in enumerate([('rear',REAR),('down',DOWN)]):
    fit,detail=observed_support(model,a[key+'_K'][k,i],mount,a[key+'_depth'][k,i],a['q'][k,i],a['gravity'][k,i],prior=None)
    info=dict(valid=fit is not None,extractor=detail)
    if fit is not None:
     fits[k,i,vi]=fit;n=np.asarray(fit[2:5],float);n/=np.linalg.norm(n)
     plane=measured(model,a['q'][k,i],a['gravity'][k,i],fit,[a['estimated_root_x'][k,i],0,0],cal)
     info.update(normal_error_deg=float(np.rad2deg(np.arccos(np.clip(n@normal,-1,1)))),offset_error_mm=float((fit[5]-offset)*1000),height_error_mm=float((plane[3]-truth['height'])*1000),slope_error_deg=float(plane[4]-truth['pitch_deg']),fit=fit.tolist())
    row['views'][key]=info
   rows.append(row)
  print('FITS_COMPLETE',float(t),flush=True)
 target=[0,1,2,3,6,7,10,11];summaries={}
 for group,subset in [('all',rows),('target', [x for x in rows if x['case'] in target]),('target_rear_tracker_unavailable',[x for x in rows if x['case'] in target and not x['recorded_rear_tracker_valid']])]:
  summaries[group]={'frames':len(subset)}
  for key in ['rear','down']:
   good=[x['views'][key] for x in subset if x['views'][key]['valid']]
   summaries[group][key]=dict(valid=len(good),invalid=len(subset)-len(good),errors={n:dict(max_abs=float(max(abs(x[n]) for x in good)),median_abs=float(np.median([abs(x[n]) for x in good]))) for n in ['normal_error_deg','offset_error_mm','height_error_mm','slope_error_deg']} if good else {})
   #Measurement-quality research check only. Not a changed physical threshold.
   summaries[group][key]['within_5mm_height_1deg_slope']=sum(abs(x['height_error_mm'])<=5 and abs(x['slope_error_deg'])<=1 for x in good)
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),diagnostic_checks_passed=True,repair_admission=False,full20s_repair_test=False,ppo_started=False,fresh_holdout=False,accounting=account,source_hashes_unchanged=True,source_hashes={str(Path(__file__).resolve()):sha(__file__),**proto['sources']},camera_mounts=poses,exact_starts=starts,prefix_comparison_to_original36=replay,actual_parent=parentcheck,fit_scope='All36times x24cases x2views independently fitted with unchanged model self-mask/conditioning/floor exclusion/largest-patch selection, prior=None. Truth labels are only error audits after selection, never choosing a plane. This is not exact replay of the live associated tracker. Rear image copied before additional down exposure; capture timing limitations are reported by measured errors.',summaries=summaries,records=rows)
 (OUT/'independent_audit.json').write_text(json.dumps(result,indent=2));np.savez_compressed(OUT/'independent_fits.npz',fits=fits,t=a['t'])
 fig,axes=plt.subplots(2,4,figsize=(15,7))
 k=int(np.argmin(abs(a['t']-12.)))
 for col,i in enumerate([0,2,6,10]):
  for vi,key in enumerate(['rear','down']):
   image=a[key+'_depth'][k,i,:,:,0];axes[vi,col].imshow(np.where(np.isfinite(image)&(image<3),image,np.nan),vmin=.1,vmax=1.5)
   row=rows[k*24+i]['views'][key];axes[vi,col].set_title(f"Case{i} {key}: fit {'valid' if row['valid'] else 'unavailable'}");axes[vi,col].set_axis_off()
 fig.suptitle('Actual recorded depth at12s — extra view does not control the robot\nUnchanged independent plane fitter; no synthetic or substituted depth');fig.tight_layout();fig.savefig(OUT/'recorded_depth.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(2,4,figsize=(15,6),sharex=True)
 for col,i in enumerate([0,2,6,10]):
  rs=[x for x in rows if x['case']==i];ts=[x['time_s'] for x in rs]
  for key in ['rear','down']:
   axes[0,col].step(ts,[x['views'][key]['valid'] for x in rs],where='post',label=key)
   axes[1,col].plot(ts,[x['views'][key].get('height_error_mm',np.nan) for x in rs],label=key)
  axes[0,col].set_title(f"{rs[0]['height_cm']:g}cm / +{rs[0]['slope_deg']:g}°");axes[1,col].set_xlabel('Time (s)')
  for ax in axes[:,col]:ax.grid(alpha=.2)
 axes[0,0].legend();axes[0,0].set_ylabel('Independent fit valid');axes[1,0].set_ylabel('Measured height error (mm)');fig.suptitle('Recorded current-view availability and diagnostic error\nNo physical repair or PPO conclusion');fig.tight_layout();fig.savefig(OUT/'availability.png',dpi=160);plt.close(fig)
 (OUT/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='SHA256.json'},indent=2))
 print(json.dumps(summaries,indent=2),flush=True)
if __name__=='__main__':main()
