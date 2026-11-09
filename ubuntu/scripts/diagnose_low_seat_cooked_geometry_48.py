"""Compare saved physical evidence against raw and queried cooked hulls.

Evaluation uses recorded poses and known development geometry only. No live
controller, source model, criteria or prior result is changed. A separate
offline model is produced for future research; it is not a runtime promotion.
"""
from pathlib import Path
import json,hashlib,datetime,copy
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from plan_seat_load_targets import clipped_vertices
from audit_low_seat_contact_probe_v2 import clean
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');OUT=D/'cooked_geometry_48'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 OUT.mkdir(exist_ok=False)
 mp=R/'runs/g1_kinematics_geometry_04/model.json';raw=BodyModel(mp);meta=read(mp);changed=copy.deepcopy(meta);query=D/'cooked_contact_v2_47'
 names=['pelvis','left_hip_pitch_link','right_hip_pitch_link','left_hip_yaw_link','right_hip_yaw_link'];hashes={str(mp):sha(mp)};cook={}
 for name in names:
  p=query/(name+'.json');r=read(p);assert r['body']==name and r['raw_points_exact_frozen_model'] and r['convex_count']==1
  cook[name]=np.concatenate([c['vertices'] for c in r['cooked_convexes']]);hashes[str(p)]=sha(p)
  records=[x for x in changed['collisions'] if x['body']==name];assert len(records)==1
  records[0]['points']=cook[name].tolist()
 # Preserve every original joint, inertial entry, default and other collider.
 replaced={x['body'] for x,y in zip(meta['collisions'],changed['collisions']) if x!=y};assert replaced==set(names)
 for k in meta:
  if k!='collisions':assert meta[k]==changed[k]
 (OUT/'offline_cooked_model.json').write_text(json.dumps(changed,indent=2))
 rows=[];curves={};variants=['normal_landing3s_06','joint_path_17','tracking_headroom_31','staged_approach_36','dual_view_approach_40']
 for variant in variants:
  folder=D/variant/'candidate';tr=np.load(folder/'trace.npz');ev=np.load(folder/'every_tick.npz');result=read(folder/'results.json');geometry=read(folder/'geometry.json');origins=np.asarray(result['origins']);bodies=result['body_names']
  for f in ['trace.npz','every_tick.npz','results.json','geometry.json']:hashes[str(folder/f)]=sha(folder/f)
  for ep in result['episodes']:
   i=ep['case'];g=geometry[ep['geometry_index']];sr=Rotation.from_quat(g['rotation_xyzw']).as_matrix();center=np.asarray(g['top_center']);hits=np.flatnonzero(ev['wrong'][:,i]>10);event=float(ev['t'][hits[0]]) if len(hits) else None
   indices={int(np.argmin(abs(tr['t']-t))) for t in [1,8,12,13,14,15,20]}
   if event is not None:indices.update(int(np.argmin(abs(tr['t']-t))) for t in [np.floor(event*10)/10,np.ceil(event*10)/10])
   if variant=='dual_view_approach_40' and i in [0,2,6,10]:indices.update(range(len(tr['t'])))
   samples=[]
   for k in sorted(indices):
    meshes=[]
    for name in ['pelvis','left_hip_pitch_link','right_hip_pitch_link']:
     pose=tr['body'][k,i,bodies.index(name)];T=transform(Rotation.from_quat(pose[3:]).as_matrix(),pose[:3]-origins[i]);gaps={}
     for label,vertices in [('raw',raw.vertices[name]),('cooked',cook[name])]:
      v=(apply(T,vertices)-center)@sr;v[:,0]-=.4;v=clipped_vertices(v);gaps[label]=float(v[:,2].min()) if len(v) else None
     meshes.append(dict(body=name,**gaps,cooked_minus_raw_m=None if None in gaps.values() else gaps['cooked']-gaps['raw']))
    geom=tr['geometry'][k,i];samples.append(dict(t=float(tr['t'][k]),seat_support_bw=float(tr['support'][k,i]),tilt_deg=float(np.rad2deg(tr['tilt'][k,i])),geometry_valid=bool(geom[11]),geometry_age_s=float(geom[12]),meshes=meshes))
   rows.append(dict(variant=variant,case=i,height_cm=g['height']*100,slope_deg=g['pitch_deg'],stable=ep['success'],clean=clean(ep),first_housing_s=event,samples=samples))
   if variant=='dual_view_approach_40' and i in [0,2,6,10]:curves[i]=rows[-1]
 summary=[]
 for r in rows:
  if r['variant']=='dual_view_approach_40' and r['case'] in [0,1,2,3,6,7,10,11]:
   time=15. if r['first_housing_s'] is None else np.floor(r['first_housing_s']*10)/10
   sample=min(r['samples'],key=lambda x:abs(x['t']-time));summary.append(dict(case=r['case'],first_housing_s=r['first_housing_s'],sample=sample))
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Recorded0.1s body poses bracket0.02s labels. All120 episodes included with fixed-time and first-event brackets; four representative dual40 cases have full sampled curves. Queried PhysX convex representation uses unchanged configured asset and installed cooking API. Not a live-shape dump or causal proof. Raw asset vertices exactly match frozen model. Contact/rest offsets were unauthored/unavailable in the query; do not substitute zero or infer force from signed gap.',source_hashes=hashes,physics_ticks=0,optimizer_updates=0,holdout_used=False,physical_admission=False,raw_model_unchanged=True,offline_model_only=str(OUT/'offline_cooked_model.json'),replaced_collision_bodies=names,rows=rows,target_summary=summary)
 (OUT/'recorded_comparison.json').write_text(json.dumps(result,indent=2))
 fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True)
 for c,i in enumerate([0,2,6,10]):
  r=curves[i];s=r['samples'];t=[x['t'] for x in s]
  for label,color in [('raw','#797979'),('cooked','#167f95')]:
   axes[0,c].plot(t,[x['meshes'][0][label]*1000 if x['meshes'][0][label] is not None else np.nan for x in s],color=color,label=label.title()+' pelvis hull')
  axes[0,c].axhline(0,color='#b54336',ls='--');axes[0,c].set_ylim(-10,30);axes[0,c].set_title(f"{r['height_cm']:g} cm / +{r['slope_deg']:g}°")
  axes[1,c].plot(t,[x['seat_support_bw'] for x in s],color='#364f85',label='Recorded legal seat load');axes[1,c].axhline(.05,color='#b54336',ls='--',label='Existing support minimum')
  if r['first_housing_s'] is not None:
   for ax in axes[:,c]:ax.axvline(r['first_housing_s'],color='#b54336',ls=':')
  for ax in axes[:,c]:ax.set_xlim(10,20);ax.grid(alpha=.2)
  axes[1,c].set_xlabel('Recorded time (s)')
 axes[0,0].set_ylabel('Pelvis gap above finite seat (mm)');axes[1,0].set_ylabel('Legal seat normal load / bodyweight');axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
 fig.suptitle('Collision cooking changes the modeled pelvis contact surface\nSaved dual-view40 evidence; dotted lines mark housing contact, not successful seating. No new physics.')
 fig.tight_layout();fig.savefig(OUT/'cooked_contact_comparison.png',dpi=150);plt.close(fig)
 # Recover metadata from already persisted valid queries; do not re-cook or rerun.
 recovery=dict(utc=result['utc'],first_attempt='Unsupported --headless rejected before script body/Kit initialization; log preserved.',second_attempt='All5 synchronous queries returned RESULT_VALID, matching raw model vertices. Timeline-stop/unchanged-time assertions precede successful exit0. Launcher close exits the process before after-context results.json could be written. Five mesh JSON outputs and log preserved; this summary reuses them, with no new query or simulation.',physics_steps=0,control_ticks=0,settling_ticks=0,optimizer_updates=0,numeric_timeline_values_not_persisted=True,mesh_hashes={str(query/(n+'.json')):sha(query/(n+'.json')) for n in names},originals={str(p):sha(p) for p in [R/'scripts/inspect_low_seat_collision_cooking_47.py',R/'scripts/inspect_low_seat_collision_cooking_v2_47.py',D/'cooked_contact_47.log',D/'cooked_contact_v2_47.log']})
 (OUT/'query_recovery_47.json').write_text(json.dumps(recovery,indent=2))
 for p,h in hashes.items():assert sha(p)==h
 (OUT/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in OUT.iterdir() if p.is_file()},indent=2));print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
