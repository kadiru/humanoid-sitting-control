"""Independent quaternion/component and sorted-statistic audit of recorded282."""
from pathlib import Path
import hashlib,json,datetime,math
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'roll_response_diagnosis_282';O=D/'roll_response_audit_283'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def matrix(q):
 q=q.astype(float);q=q/np.sqrt((q*q).sum(axis=-1))[...,None];x,y,z,w=np.moveaxis(q,-1,0)
 return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(*q.shape[:-1],3,3)
def stats(x):
 s=sorted(float(v) for v in np.asarray(x).ravel());n=len(s);return dict(min=s[0],max=s[-1],median=(s[(n-1)//2]+s[n//2])/2,rms=math.sqrt(math.fsum(v*v for v in s)/n),range=s[-1]-s[0])
def main():
 O.mkdir(exist_ok=False);r=read(P/'results.json');a=dict(np.load(P/'paired_arrays.npz'));assert all(sha(p)==h for p,h in r['source_sha256'].items());model=read(D/'cooked_model_50/model.json');ids=r['joint_ids'];bi=r['body_ids'];geom=0.;checks=0;staterr=0.;traces=[]
 for si,name in enumerate(['centered_probe_234','roll_feedback_probe_273']):
  C=D/name/'candidate';tr=dict(np.load(C/'trace.npz'));traces.append(tr);ticks=dict(np.load(C/'fixed_ticks_132.npz'));ev=dict(np.load(C/'every_tick.npz'));raw={(x['case'],x['tick']):x for x in read(C/'touchdown_raw_176.json')}
  assert np.array_equal(a['t'],tr['t']) and np.array_equal(a['command_t'],ticks['t'])
  assert np.max(abs(ticks['t']-(np.arange(1000)+1)*.02))<1e-12
  for name in ['q','command','torque']:assert np.array_equal(a[name][si],tr[name])
  assert np.array_equal(a['every_command'][si],ticks['after']);assert np.array_equal(a['body_xyz'][si],tr['body'][:,:,bi,:3])
  R=matrix(tr['body'][:,:,bi,3:]);geom=max(geom,float(abs(R-a['body_R'][si]).max()));assert geom<1e-12
  roll=np.arctan2(R[...,2,1],R[...,2,2]);pitch=np.arctan2(-R[...,2,0],np.sqrt(R[...,2,1]**2+R[...,2,2]**2))
  assert np.max(abs(roll-a['body_roll'][si]))<1e-12 and np.max(abs(pitch-a['body_pitch'][si]))<1e-12
  for side,s in enumerate(['left','right']):
   points=next(x['points'] for x in model['collisions'] if x['body']==s+'_ankle_roll_link')
   for v,p in enumerate(points):
    for j in range(3):
     expected=tr['body'][:,:,bi[side+1],j].astype(float)+sum(R[:,:,side+1,j,k]*p[k] for k in range(3));geom=max(geom,float(np.max(abs(expected-a['foot_vertices'][si,:,:,side,v,j]))));checks+=expected.size
   assert np.array_equal(a['foot_load_bw'][si,:,:,side],ev['force'][:,:,side+5,2]/ev['weight'])
  for i,t in enumerate(a['raw_t']):
   tick=int(round(t/.02))
   for c in [0,1]:
    for key,field in [('raw_q','q'),('raw_qd','qd'),('raw_gyro','gyro')]:assert np.array_equal(a[key][si,i,c],raw[c,tick][field])
 assert geom<1e-11
 def check(x,y):
  nonlocal staterr
  z=stats(x)
  for k in z:staterr=max(staterr,abs(z[k]-y[k]))
 for row in r['rows']:
  c=row['case'];lo,hi=row['window'];m=(a['t']>=lo)&(a['t']<=hi);n=(a['command_t']>=lo)&(a['command_t']<=hi);assert row['trace_samples']==int(m.sum()) and row['command_ticks']==int(n.sum())
  for j,saved in zip(ids,row['joints']):
   for key in ['q','command','torque']:check(a[key][1,m,c,j]-a[key][0,m,c,j],saved['paired_difference'][key])
   for si,label in [(0,'old_q'),(1,'new_q')]:check(a['q'][si,m,c,j],saved[label])
  for b,saved in enumerate(row['bodies']):
   check(a['body_roll'][1,m,c,b]-a['body_roll'][0,m,c,b],saved['paired_roll_rad']);check(a['body_pitch'][1,m,c,b]-a['body_pitch'][0,m,c,b],saved['paired_pitch_rad'])
   for j in range(3):check(a['body_xyz'][1,m,c,b,j]-a['body_xyz'][0,m,c,b,j],saved['paired_xyz_m'][j])
  for side in range(2):
   for si,label in [(0,'old'),(1,'new')]:
    check(a['foot_load_bw'][si,n,c,side],row['actual_foot_load_bw'][side][label]);check(a['foot_vertices'][si,m,c,side,:,2].min(axis=-1),row['cooked_foot_bottom_z_m'][side][label])
  if c in [0,1]:
   m=(a['raw_t']>=lo)&(a['raw_t']<=hi)
   for side,j in enumerate(ids):
    z=row['raw_joints'][side];check(a['raw_qd'][0,m,c,j],z['old_qd']);check(a['raw_qd'][1,m,c,j],z['new_qd']);check(a['raw_qd'][1,m,c,j]-a['raw_qd'][0,m,c,j],z['paired_qd']);check(a['roll_feedback'][m,c,side],z['offset'])
 assert staterr<1e-11
 def first(t,x):return next((float(u) for u,v in zip(t,x) if v),None)
 for z in r['onsets']:
  c=z['case'];assert z['first_command_difference_s']==first(a['command_t'],np.any(a['every_command'][1,:,c]!=a['every_command'][0,:,c],axis=-1));assert z['first_sampled_body_difference_s']==first(a['t'],np.any(traces[1]['body'][:,c]!=traces[0]['body'][:,c],axis=(1,2)));assert z['first_sampled_geometry_difference_s']==first(a['t'],np.any(traces[1]['geometry'][:,c]!=traces[0]['geometry'][:,c],axis=-1))
  if c in [0,1]:
   assert z['first_raw_q_difference_s']==first(a['raw_t'],np.any(a['raw_q'][1,:,c]!=a['raw_q'][0,:,c],axis=-1));assert z['roll_feedback_onset_s']==first(a['raw_t'],np.any(a['roll_feedback'][:,c]!=0,axis=-1))
 b={(x['case'],x['tick']):x for x in read(D/'roll_feedback_probe_273/candidate/balance_feedback_161.json')}
 for i,t in enumerate(a['raw_t']):
  for c in [0,1]:
   v=b[c,round(t/.02)]['feedback']['roll'];assert np.array_equal(a['roll_feedback'][i,c],v['after']) and a['roll_enabled'][i,c]==v['enabled']
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,source_arrays_exact=True,geometry_component_values=checks,max_geometry_error=geom,case_windows=len(r['rows']),max_sorted_statistic_error=staterr,all24_onsets_checked=True,physics_ticks=0,optimizer_updates=0,physical_admission=False,clock_note='A feedback record at tickk describes a command applied over [k*.02,(k+1)*.02]. fixed_ticks after command carries interval-end time. Onset9.38 versus9.40 is not evidence of20ms implementation latency. Raw q at9.4 follows that interval;100ms body observations only bracket motion.',source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),P/'results.json',P/'paired_arrays.npz']})
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
