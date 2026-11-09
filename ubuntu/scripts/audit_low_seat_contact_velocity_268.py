"""Independent vector/matrix replay of267, using all saved normal contacts."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'contact_wrench_diagnostic_259';I=D/'contact_velocity_diagnosis_267';O=D/'contact_velocity_audit_268'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);r=read(I/'results.json');assert all(sha(p)==h for p,h in r['source_sha256'].items())
 a=dict(np.load(I/'contact_points.npz'));s=np.load(P/'substep_states.npz');meta=read(P/'actuator_metadata.json');inertia=read(P/'static_inertial_model.json')
 poses=s['link_transforms'].astype(float);v=s['link_velocities'].astype(float);R=Rotation.from_quat(poses[...,3:].reshape(-1,4)).as_matrix().reshape(*poses.shape[:3],3,3)
 # Reconstruct the retained contact list directly from original returned buffers.
 cr=np.load(P/'contact_returns.npz');cm=read(P/'contact_recorder_metadata.json');points=[];seps=[];normals=[];forces=[]
 for i,e in enumerate(cm['records']):
  if e['kind']!='normal' or e['substep']==1600:continue
  pp=cr[f'r{i}_a1'];nn=cr[f'r{i}_a2'];ss=cr[f'r{i}_a3'];ff=cr[f'r{i}_a0'];counts=cr[f'r{i}_a4'];starts=cr[f'r{i}_a5']
  for c in range(24):
   ix=np.arange(int(starts[c,0]),int(starts[c,0])+int(counts[c,0]));points.extend(pp[ix]);normals.extend(nn[ix]);seps.extend(ss[ix,0]);forces.extend(ff[ix,0])
 for name,value in [('point',points),('normal',normals),('separation',seps),('normal_force',forces)]:assert np.array_equal(a[name],np.asarray(value)),name
 k=a['substep'].astype(int)-1600;c=a['case'].astype(int);side=a['side'].astype(int);bs=np.array([meta['body_names'].index(x+'_ankle_roll_link') for x in ['left','right']]);b=bs[side]
 coms=np.array([inertia['local_com'][inertia['body_names'].index(x+'_ankle_roll_link')] for x in ['left','right']])[side]
 com=poses[k,c,b,:3]+np.einsum('nij,nj->ni',R[k,c,b],coms);radius=a['point']-com;w=v[k,c,b,3:];linear=v[k,c,b,:3]
 # Component expansion independent of np.cross used in267.
 cross=np.column_stack((w[:,1]*radius[:,2]-w[:,2]*radius[:,1],w[:,2]*radius[:,0]-w[:,0]*radius[:,2],w[:,0]*radius[:,1]-w[:,1]*radius[:,0]));pv=linear+cross
 relative=np.einsum('nij,nkj->nik',R[k-1,c,b],R[k,c,b]);prev=poses[k-1,c,b,:3]+np.einsum('nij,nj->ni',relative,a['point']-poses[k,c,b,:3]);fd=(a['point']-prev)/.005
 vn=(pv*a['normal']).sum(1);tangent=np.sqrt(((pv-vn[:,None]*a['normal'])**2).sum(1));vfd=(fd*a['normal']).sum(1)
 errors={name:float(abs(value-a[name]).max()) for name,value in [('com',com),('point_velocity',pv),('material_point_fd',fd),('normal_velocity',vn),('tangent_speed',tangent),('normal_fd',vfd)]}
 assert max(errors.values())<1e-11,errors
 for row in r['rows']:
  mask=(c==row['case'])&(side==['left','right'].index(row['side']))&(k>=600);assert mask.sum()==row['points']
  for name in ['normal_velocity','tangent_speed','normal_fd','separation','normal_force','qd_ankle_roll']:
   x=a[name][mask];z=dict(min=float(x.min()),median=float(np.median(x)),max=float(x.max()),rms=float(np.sqrt((x*x).mean())))
   assert z==row[name]
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,points=len(c),rows=len(r['rows']),maximum_errors=errors,arithmetic_limit=1e-11,physics_ticks=0,optimizer_updates=0,physical_admission=False,raw_velocity_guard_unchanged=True,source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),I/'results.json',I/'contact_points.npz',P/'contact_returns.npz',P/'substep_states.npz',P/'static_inertial_model.json']})
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
