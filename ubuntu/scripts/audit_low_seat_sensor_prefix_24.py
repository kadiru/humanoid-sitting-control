"""Audit the bounded8.4s physical sensor diagnostic, without new simulation."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel
from humanoid_sitting.supported_surface import observed_support
from run_supported_postures import REAR_CAMERA
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'sensor_prefix_24'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def comparison(a,b):
 finite=np.isfinite(a)&np.isfinite(b);same=(a==b)|(np.isnan(a)&np.isnan(b))
 return dict(exact=bool(same.all()),changed=int((~same).sum()),max_finite_difference=float(abs(a[finite]-b[finite]).max()) if finite.any() else None,finite_pattern_equal=bool(np.array_equal(np.isfinite(a),np.isfinite(b))))
def main():
 assert not (P/'independent_audit.json').exists()
 states={};existing={};replay={};selected=[0,1,6,7,10,11];unselected=[i for i in range(24) if i not in selected]
 for name,old in [('baseline',D/'normal_landing3s_06/candidate'),('changed',D/'joint_path_17/candidate')]:
  account=read(P/name/'diagnostic_accounting.json');assert account['active_transitions']==10080 and account['initial_settling_excluded']==600 and account['full20s_episodes']==0
  states[name]=dict(np.load(P/name/'raw_sensor_captures.npz'));existing[name]=dict(np.load(old/'trace.npz'))
  new=dict(np.load(P/name/'trace.npz'));original=existing[name]
  replay[name]={k:comparison(new[k],original[k][:len(new[k])]) for k in ['t','body','q','command','geometry','reference','torque']}
 A=states['baseline'];B=states['changed'];assert np.array_equal(A['t'],B['t'])
 m=BodyModel('runs/g1_kinematics_geometry_04/model.json');out=[];refits=[]
 for k,t in enumerate(A['t']):
  own={name:comparison(A[name][k,unselected],B[name][k,unselected]) for name in ['q','gravity','body','K','camera_position','camera_orientation','depth','geometry']}
  depth=abs(A['depth'][k].astype(float)-B['depth'][k]);finite=np.isfinite(A['depth'][k])&np.isfinite(B['depth'][k]);dd=np.where(finite,depth,0)
  percase=[]
  for i in unselected:
   prior_at=max(0,np.flatnonzero(existing['baseline']['t']<t)[-1]);prior=existing['baseline']['geometry'][prior_at,i]
   ga,da=observed_support(m,A['K'][k,i],REAR_CAMERA,A['depth'][k,i],A['q'][k,i],A['gravity'][k,i],prior)
   gb,db=observed_support(m,B['K'][k,i],REAR_CAMERA,B['depth'][k,i],B['q'][k,i],B['gravity'][k,i],prior)
   percase.append(dict(case=i,depth= comparison(A['depth'][k,i],B['depth'][k,i]),geometry=comparison(A['geometry'][k,i],B['geometry'][k,i]),depth_changed_finite_above_1um=int((dd[i]>1e-6).sum()),depth_changed_finite_above_1mm=int((dd[i]>.001).sum()),refit_valid=ga is not None and gb is not None,refit_difference=comparison(ga,gb) if ga is not None and gb is not None else None,baseline_refit_to_recorded=comparison(ga,A['geometry'][k,i]) if ga is not None else None,changed_refit_to_recorded=comparison(gb,B['geometry'][k,i]) if gb is not None else None))
  out.append(dict(time=float(t),unselected=own,per_case=percase))
  if abs(t-8.2)<1e-6:
   import matplotlib
   matplotlib.use('Agg');import matplotlib.pyplot as plt
   fig,axes=plt.subplots(1,3,figsize=(12,3.5));i=3
   axes[0].imshow(np.where(np.isfinite(A['depth'][k,i,:,:,0]),A['depth'][k,i,:,:,0],np.nan),vmin=.3,vmax=2.5);axes[0].set_title('Baseline depth — unselected case3')
   axes[1].imshow(np.where(np.isfinite(B['depth'][k,i,:,:,0]),B['depth'][k,i,:,:,0],np.nan),vmin=.3,vmax=2.5);axes[1].set_title('Selected neighbours changed')
   im=axes[2].imshow(dd[i,:,:,0]*1e6,cmap='magma');axes[2].set_title('Absolute finite depth difference (µm)');fig.colorbar(im,ax=axes[2])
   for ax in axes:ax.set_axis_off()
   fig.suptitle('Matched8.2s diagnostic frames; not a repair evaluation');fig.tight_layout();fig.savefig(P/'depth_comparison.png',dpi=160);plt.close(fig)
 proto=read(P/'protocol.json');hashes_ok=all(sha(p)==h for p,h in proto['source_hashes'].items())
 d=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),completed=True,physical_repair_evaluation=False,ppo_updates=0,fresh_holdout=False,total_active_transitions=20160,initial_settling_excluded=1200,source_hashes_unchanged=hashes_ok,replay_to_original=replay,records=out,scope='Only8.4s diagnostic prefixes. Own inputs/poses, raw rendered depth and independent plane refits distinguish sensor-scene coupling from controller mutation. Full20s physical repair and effort remain untested; original joint_path17 is still rejected.')
 (P/'independent_audit.json').write_text(json.dumps(d,indent=2)+'\n');print(json.dumps({k:v for k,v in d.items() if k!='records'},indent=2))
 for r in out:print(r['time'],r['unselected'])
if __name__=='__main__':main()
