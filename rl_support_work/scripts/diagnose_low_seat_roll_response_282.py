"""Saved234/273 paired response and foot geometry; no new physics or fitted controller."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial.transform import Rotation
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'roll_response_diagnosis_282'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stat(a):
 a=np.asarray(a,float);return dict(min=float(a.min()),median=float(np.median(a)),max=float(a.max()),rms=float(np.sqrt(np.mean(a*a))),range=float(np.ptp(a)))
def first(t,x):
 k=np.flatnonzero(x);return float(t[k[0]]) if len(k) else None
def main():
 O.mkdir(exist_ok=False);paths=[D/n/'candidate' for n in ['centered_probe_234','roll_feedback_probe_273']];sources=[Path(__file__).resolve(),D/'cooked_model_50/model.json'];model=read(sources[-1]);bodies=read(paths[0]/'results.json')['body_names'];names=read(paths[0]/'results.json')['joint_names'];ids=[names.index(s+'_ankle_roll_joint') for s in ['left','right']];bi=[bodies.index(s) for s in ['pelvis','left_ankle_roll_link','right_ankle_roll_link']]
 tr=[];ticks=[];raw=[];balance=[];events=[];foot=[];rot=[]
 for p in paths:
  sources.extend(p/f for f in ['trace.npz','fixed_ticks_132.npz','touchdown_raw_176.json','balance_feedback_161.json','every_tick.npz','results.json'])
  tr.append(dict(np.load(p/'trace.npz')));ticks.append(dict(np.load(p/'fixed_ticks_132.npz')));raw.append({(r['case'],r['tick']):r for r in read(p/'touchdown_raw_176.json')});balance.append({(r['case'],r['tick']):r for r in read(p/'balance_feedback_161.json')});events.append(dict(np.load(p/'every_tick.npz')))
  q=tr[-1]['body'][:,:,bi,3:].astype(float);R=Rotation.from_quat(q.reshape(-1,4)).as_matrix().reshape(*q.shape[:-1],3,3);rot.append(R)
  feet=[]
  for i,s in enumerate(['left','right']):
   v=np.array(next(x for x in model['collisions'] if x['body']==s+'_ankle_roll_link')['points']);assert v.shape==(8,3)
   feet.append(np.einsum('tcij,vj->tcvi',R[:,:,i+1],v)+tr[-1]['body'][:,:,bi[i+1],None,:3])
  foot.append(np.stack(feet,axis=2))
 assert np.array_equal(tr[0]['t'],tr[1]['t']) and np.array_equal(ticks[0]['t'],ticks[1]['t'])
 t=tr[0]['t'];rt=np.array(sorted({k[1] for k in raw[0]}));assert set(raw[0])==set(raw[1]);assert len(rt)==950
 a=dict(t=t,command_t=ticks[0]['t'],raw_t=rt*.02,q=np.stack([x['q'] for x in tr]),command=np.stack([x['command'] for x in tr]),torque=np.stack([x['torque'] for x in tr]),every_command=np.stack([x['after'] for x in ticks]),body_xyz=np.stack([x['body'][:,:,bi,:3] for x in tr]),body_R=np.stack(rot),foot_vertices=np.stack(foot),raw_q=np.array([[[r[c,k]['q'] for c in [0,1]] for k in rt] for r in raw]),raw_qd=np.array([[[r[c,k]['qd'] for c in [0,1]] for k in rt] for r in raw]),raw_gyro=np.array([[[r[c,k]['gyro'] for c in [0,1]] for k in rt] for r in raw]),roll_feedback=np.array([[balance[1][c,k]['feedback']['roll']['after'] for c in [0,1]] for k in rt]),roll_enabled=np.array([[balance[1][c,k]['feedback']['roll']['enabled'] for c in [0,1]] for k in rt]))
 # z-axis tilt and lateral slope describe link orientation, not a live collision certificate.
 a['body_roll']=np.arctan2(a['body_R'][...,2,1],a['body_R'][...,2,2]);a['body_pitch']=np.arctan2(-a['body_R'][...,2,0],np.hypot(a['body_R'][...,2,1],a['body_R'][...,2,2]))
 a['foot_load_bw']=np.stack([e['force'][:,:,5:7,2]/e['weight'][:,:,None] for e in events]);assert all(np.isfinite(v).all() for v in a.values());np.savez_compressed(O/'paired_arrays.npz',**a)
 rows=[]
 for c in range(24):
  for lo,hi in [(8,9.2),(9.4,11),(11,14.9),(15,20)]:
   mask=(t>=lo)&(t<=hi);wm=(a['command_t']>=lo)&(a['command_t']<=hi);row=dict(case=c,window=[lo,hi],trace_samples=int(mask.sum()),command_ticks=int(wm.sum()),joints=[],bodies=[])
   for j in ids:
    d={label:stat(a[label][1,mask,c,j]-a[label][0,mask,c,j]) for label in ['q','command','torque']};row['joints'].append(dict(joint=names[j],paired_difference=d,old_q=stat(a['q'][0,mask,c,j]),new_q=stat(a['q'][1,mask,c,j])))
   for b,name in enumerate(['pelvis','left_ankle_roll_link','right_ankle_roll_link']):
    row['bodies'].append(dict(body=name,paired_roll_rad=stat(a['body_roll'][1,mask,c,b]-a['body_roll'][0,mask,c,b]),paired_pitch_rad=stat(a['body_pitch'][1,mask,c,b]-a['body_pitch'][0,mask,c,b]),paired_xyz_m=[stat(a['body_xyz'][1,mask,c,b,j]-a['body_xyz'][0,mask,c,b,j]) for j in range(3)]))
   row['actual_foot_load_bw']=[dict(old=stat(a['foot_load_bw'][0,wm,c,j]),new=stat(a['foot_load_bw'][1,wm,c,j])) for j in range(2)]
   row['cooked_foot_bottom_z_m']=[dict(old=stat(a['foot_vertices'][0,mask,c,j,:,2].min(axis=-1)),new=stat(a['foot_vertices'][1,mask,c,j,:,2].min(axis=-1))) for j in range(2)]
   if c in [0,1]:
    rm=(a['raw_t']>=lo)&(a['raw_t']<=hi);row['raw_joints']=[dict(joint=names[j],old_qd=stat(a['raw_qd'][0,rm,c,j]),new_qd=stat(a['raw_qd'][1,rm,c,j]),paired_qd=stat(a['raw_qd'][1,rm,c,j]-a['raw_qd'][0,rm,c,j]),offset=stat(a['roll_feedback'][rm,c,k])) for k,j in enumerate(ids)]
   rows.append(row)
 onsets=[]
 for c in range(24):
  onsets.append(dict(case=c,first_command_difference_s=first(a['command_t'],np.any(a['every_command'][1,:,c]!=a['every_command'][0,:,c],axis=-1)),first_sampled_body_difference_s=first(t,np.any(tr[1]['body'][:,c]!=tr[0]['body'][:,c],axis=(1,2))),first_sampled_geometry_difference_s=first(t,np.any(tr[1]['geometry'][:,c]!=tr[0]['geometry'][:,c],axis=-1)),roll_feedback_onset_s=first(a['raw_t'],np.any(a['roll_feedback'][:,c]!=0,axis=-1)) if c in [0,1] else None,first_raw_q_difference_s=first(a['raw_t'],np.any(a['raw_q'][1,:,c]!=a['raw_q'][0,:,c],axis=-1)) if c in [0,1] else None))
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,joint_names=names,body_names=bodies,selected_cases=[0,1],joint_ids=ids,body_ids=bi,rows=rows,onsets=onsets,physical_admission=False,repair_complete=False,physics_ticks=0,optimizer_updates=0,source_sha256={str(p):sha(p) for p in sources},limitations=['Feedback is endogenous; paired differences/onset timing do not identify a causal transfer function or justify gain extrapolation.','Raw precontrol q/qd and postcontrol100ms trace are separate recorded clocks; no interpolation/alignment fit or finite-difference substitution.','Actual body/contact loads are evaluation only. Cooked vertices are offline geometry with unknown live contact/rest offsets.','Approximate implicit-PD torque telemetry is not actual drive force; endstate PD reconstruction is intentionally not used.','All24 cases and both full20s runs included; no new episode or physical/quiet criterion change.'])
 (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2));print(json.dumps(dict(records=len(rows),selected_onsets=onsets[:2],selected_late=[r for r in rows if r['case'] in [0,1] and r['window']==[11,14.9]])),flush=True)
if __name__=='__main__':main()
