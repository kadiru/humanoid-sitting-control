"""Offline contact-point velocity from verified259 records; no controller change."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'contact_wrench_diagnostic_259';O=D/'contact_velocity_diagnosis_267'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(v):return dict(min=float(np.min(v)),median=float(np.median(v)),max=float(np.max(v)),rms=float(np.sqrt(np.mean(np.square(v))))) if len(v) else None
def main():
 O.mkdir(exist_ok=False);assert read(D/'contact_wrench_audit_260/independent_audit.json')['exact_prefix_replay']
 s=np.load(P/'substep_states.npz');pose=s['link_transforms'].astype(float);vel=s['link_velocities'].astype(float);qd=s['qd'];t=s['t']
 meta=read(P/'actuator_metadata.json');inertia=read(P/'static_inertial_model.json');cm=read(P/'contact_recorder_metadata.json');cr=np.load(P/'contact_returns.npz')
 R=Rotation.from_quat(pose[:,:,:,3:].reshape(-1,4)).as_matrix().reshape(*pose.shape[:3],3,3)
 localcom={b:np.asarray(c) for b,c in zip(inertia['body_names'],inertia['local_com'])};assert all(np.isfinite(v).all() for v in localcom.values())
 data={k:[] for k in ['case','side','substep','point','normal','normal_force','separation','com','link_linear','link_angular','point_velocity','material_point_fd','normal_velocity','tangent_speed','normal_fd','qd_ankle_roll']}
 for i,r in enumerate(cm['records']):
  if r['kind']!='normal':continue
  k=r['substep']-1600
  if k==0:continue # No prior private pose for the same material-point difference.
  side=r['side'];b=meta['body_names'].index(side+'_ankle_roll_link');j=meta['joint_names'].index(side+'_ankle_roll_joint');c0=localcom[side+'_ankle_roll_link']
  a=[cr[f'r{i}_a{j}'] for j in range(6)]
  for c in range(24):
   start=int(a[5][c,0]);n=int(a[4][c,0]);sl=slice(start,start+n)
   if not n:continue
   p=a[1][sl].astype(float);normal=a[2][sl].astype(float);f=a[0][sl,0].astype(float);sep=a[3][sl,0].astype(float)
   origin=pose[k,c,b,:3];rot=R[k,c,b];com=origin+rot@c0;lv=vel[k,c,b,:3];omega=vel[k,c,b,3:]
   v=lv+np.cross(omega,p-com);local=(p-origin)@rot
   prev=local@R[k-1,c,b].T+pose[k-1,c,b,:3];fd=(p-prev)/.005
   vn=np.sum(v*normal,axis=1);vf=np.sum(fd*normal,axis=1);tan=np.linalg.norm(v-vn[:,None]*normal,axis=1)
   vals=dict(case=np.full(n,c),side=np.full(n,0 if side=='left' else 1),substep=np.full(n,r['substep']),point=p,normal=normal,normal_force=f,separation=sep,com=np.tile(com,(n,1)),link_linear=np.tile(lv,(n,1)),link_angular=np.tile(omega,(n,1)),point_velocity=v,material_point_fd=fd,normal_velocity=vn,tangent_speed=tan,normal_fd=vf,qd_ankle_roll=np.full(n,qd[k,c,j]))
   for name,x in vals.items():data[name].append(x)
 a={k:np.concatenate(v) for k,v in data.items()};assert all(np.isfinite(v).all() for v in a.values());np.savez_compressed(O/'contact_points.npz',**a)
 rows=[]
 for c in range(24):
  for side in range(2):
   mask=(a['case']==c)&(a['side']==side)&(a['substep']>=2200)
   row=dict(case=c,side=['left','right'][side],points=int(mask.sum()),loaded_points=int((mask&(a['normal_force']>0)).sum()))
   for name in ['normal_velocity','tangent_speed','normal_fd','separation','normal_force','qd_ankle_roll']:row[name]=stats(a[name][mask])
   rows.append(row)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,points=len(a['case']),window_s=[11,13],rows=rows,physics_ticks=0,optimizer_updates=0,physical_admission=False,source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),P/'substep_states.npz',P/'contact_returns.npz',P/'contact_recorder_metadata.json',P/'actuator_metadata.json',P/'static_inertial_model.json',D/'contact_wrench_audit_260/independent_audit.json']},velocity_frame='Installed API: global linear velocity at link COM. Use recorded259 local COM, not planner bridge inertia. Contact-plane normals are recorded labels.',interpretation='Point velocities use poststep backend twist and same saved contact positions. Finite differences follow the same material point through adjacent poses, not changing contact point identity. These remain different sampling quantities. No substitution, solver-defect claim, causality or guard waiver.',documentation_url='https://docs.omniverse.nvidia.com/kit/docs/omni_physics/109.0/extensions/runtime/source/omni.physics.tensors/docs/api/python.html#omni.physics.tensors.impl.api.ArticulationView.get_link_velocities')
 (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2));print(json.dumps(dict(points=out['points'],selected=rows[:4])),flush=True)
if __name__=='__main__':main()
