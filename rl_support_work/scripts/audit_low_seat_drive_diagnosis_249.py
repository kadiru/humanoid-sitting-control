"""Independent scalar algebra and quaternion-vector replay; zero physics."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'drive_diagnosis_audit_249'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rotate(q,v):
 q=np.asarray(q,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True);u=q[...,:3];return v+2*q[...,3,None]*np.cross(u,v)+2*np.cross(u,np.cross(u,v))
def st(x):return dict(min=float(np.min(x)),max=float(np.max(x)),median=float(np.median(x)),mean=float(np.mean(x)))
def main():
 O.mkdir(exist_ok=False);a=read(D/'roll_drive_diagnosis_243/results.json');b=read(D/'drive_diagnosis_248/results.json')
 for result in [a,b]:assert all(sha(p)==h for p,h in result['source_sha256'].items())
 meta=read(D/'drive_diagnostic_246/actuator_metadata.json');joints={x['name']:x for x in read(D/'cooked_model_50/model.json')['joints']};s=np.load(D/'drive_diagnostic_246/substep_states.npz');replay=np.load(D/'drive_diagnosis_248/velocity_closure.npz');err=0.;count=0
 for j,n in enumerate(meta['joint_names']):
  joint=joints[n];p=meta['body_names'].index(joint['body0'][0]);c=meta['body_names'].index(joint['body1'][0]);basis=np.eye(3)['XYZ'.index(joint['axis'])];axis=rotate(s['link_transforms'][:,:,p,3:],rotate(joint['q0'],basis));rel=s['link_velocities'][:,:,c,3:].astype(float)-s['link_velocities'][:,:,p,3:].astype(float);proj=np.sum(axis*rel,axis=-1);res=proj-s['qd'][:,:,j];off=np.sqrt(np.sum((rel-proj[...,None]*axis)**2,axis=-1));err=max(err,float(abs(res-replay['joint_link_residual'][:,:,j]).max()),float(abs(off-replay['off_axis_speed'][:,:,j]).max()));count+=res.size
 assert err<1e-12
 cache={};maxstat=0.;origmeta=read(D/'private_diagnostic_198/actuator_metadata.json')
 for r in a['rows']:
  if r['run'] not in cache:
   P=D/r['run']/'candidate';cache[r['run']]=(dict(np.load(P/'trace.npz')),np.load(P/'sensor_inputs.npz')['sensors'][0])
  tr,frames=cache[r['run']];c=r['case'];j=r['joint'];k=np.arange(round(r['window_s'][0]*10),round(r['window_s'][1]*10)+1);kp=meta['stiffness'][c][j];kd=meta['damping'][c][j]
  error=tr['command'][k-1,c,j]-tr['q'][k-1,c,j];velocity=frames[k,c,293+j]*np.float32(10)
  # Preserve the original float32 subtraction/multiplication order, then
  # compare independently selected scalar series statistics.
  pterm=np.float64(origmeta['stiffness'][c][j])*error;dterm=-np.float64(origmeta['damping'][c][j])*velocity
  for key,series in [('raw_velocity_rad_s',velocity),('command_error_rad',error),('position_term_Nm',pterm),('velocity_term_Nm',dterm)]:
   for name,value in st(series).items():maxstat=max(maxstat,abs(value-r[key][name]))
 assert maxstat<1e-12
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,geometry_velocity_values=count,quaternion_replay_max_error_rad_s=err,drive_records_replayed=len(a['rows']),drive_statistics_max_error=maxstat,source_hashes_unchanged=True,physical_admission=False,physics_ticks=0,optimizer_updates=0,interpretation='Arithmetic audit only. Large recorded joint/link velocity disagreement, if present, is preserved, not rounded away or relabeled physical success.',source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),D/'roll_drive_diagnosis_243/results.json',D/'drive_diagnosis_248/results.json',D/'drive_diagnosis_248/velocity_closure.npz']})
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({'independent_audit.json':sha(O/'independent_audit.json')},indent=2));print(json.dumps(out))
if __name__=='__main__':main()
