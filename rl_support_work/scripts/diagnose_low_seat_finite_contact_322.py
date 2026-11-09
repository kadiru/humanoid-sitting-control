"""Finite contact-return assessment. Offline evaluation only; no forward predictor."""
from pathlib import Path
import json, hashlib, datetime
import numpy as np
from scipy.spatial.transform import Rotation
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01'); P=D/'contact_wrench_diagnostic_259'; O=D/'finite_contact_diagnosis_322'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def stats(x):
 x=np.asarray(x); x=x[np.isfinite(x)]
 return None if not len(x) else dict(count=len(x),min=float(x.min()),median=float(np.median(x)),max=float(x.max()),rms=float(np.sqrt(np.mean(x*x))))
def main():
 assert read(D/'contact_wrench_audit_260/independent_audit.json')['exact_prefix_replay']
 assert read(D/'contact_velocity_audit_268/independent_audit.json')['passed']
 scope=read(D/'finite_contact_revision_322.json'); O.mkdir(exist_ok=False)
 a=dict(np.load(D/'contact_velocity_diagnosis_267/contact_points.npz')); s=dict(np.load(P/'substep_states.npz'))
 meta=read(P/'actuator_metadata.json'); inertia=read(P/'static_inertial_model.json'); cm=read(P/'contact_recorder_metadata.json'); cr=np.load(P/'contact_returns.npz')
 coverage=np.load(D/'ankle_impulse_296/arrays.npz')['coverage']
 assert coverage.shape==(1000,24)
 R=Rotation.from_quat(s['link_transforms'][...,3:].reshape(-1,4)).as_matrix().reshape(1001,24,44,3,3)
 body=[meta['body_names'].index(z+'_ankle_roll_link') for z in ['left','right']]
 comlocal=np.array([inertia['local_com'][inertia['body_names'].index(z+'_ankle_roll_link')] for z in ['left','right']])
 # Group all original normal records, retaining points with exactly zero force.
 gid=((a['substep'].astype(int)-1601)*24+a['case'].astype(int))*2+a['side'].astype(int)
 order=np.argsort(gid,kind='stable'); counts=np.bincount(gid,minlength=48000); starts=np.r_[0,np.cumsum(counts)]
 friction={}
 for i,r in enumerate(cm['records']):
  if r['kind']=='friction' and r['substep']>1600:
   assert r['dt']==.005
   friction[(r['substep'],0 if r['side']=='left' else 1)]=[cr[f'r{i}_a{j}'] for j in range(4)]
 rows=[]; singular=[]; normal_points=0; friction_points=0
 for k in range(1,1001):
  for c in range(24):
   for side in range(2):
    g=((k-1)*24+c)*2+side; ix=order[starts[g]:starts[g+1]]
    f=a['normal_force'][ix]; n=a['normal'][ix]; p=a['point'][ix]; vn=a['normal_velocity'][ix]; fd=a['normal_fd'][ix]; sep=a['separation'][ix]
    normal_points+=len(ix); assert (f>=0).all()
    F=float(f.sum()); loaded=f>0; denom=F if F>0 else np.nan
    b=body[side]; pose=s['link_transforms'][k,c,b]; rot=R[k,c,b]; center=pose[:3]+rot@comlocal[side]
    # A normal point supplies one scalar row, not six bilateral foot constraints.
    C=np.c_[n,np.cross(p-center,n)/.1][loaded]
    sv=np.linalg.svd(C,compute_uv=False) if len(C) else np.array([])
    singular.append(np.pad(sv,(0,6-len(sv))))
    rank=int((sv>1e-10).sum()) # numerical rank only, never a physical tolerance
    ff,fp,fc,fs=friction[(k+1600,side)]; j=int(fs[c,0]); nf=int(fc[c,0]); fv=ff[j:j+nf].astype(float); pp=fp[j:j+nf].astype(float); friction_points+=nf
    vel=s['link_velocities'][k,c,b].astype(float); pv=vel[:3]+np.cross(vel[3:],pp-center)
    local=(pp-pose[:3])@rot; prev=local@R[k-1,c,b].T+s['link_transforms'][k-1,c,b,:3]; pfd=(pp-prev)/.005
    norms=np.linalg.norm(fv,axis=1)
    row=dict(substep=k+1600,case=c,side=side,observed_floor_only=int(coverage[k-1,c]),normal_points=len(ix),loaded_points=int(loaded.sum()),friction_points=nf,normal_rank=rank,normal_force=F,
      normal_impulse=F*.005,positive_velocity_force=float(f[vn>0].sum()),negative_velocity_force=float(f[vn<0].sum()),zero_velocity_force=float(f[vn==0].sum()),positive_gap_force=float(f[sep>0].sum()),
      weighted_normal_velocity=float(np.dot(f,vn)/denom),weighted_abs_normal_velocity=float(np.dot(f,abs(vn))/denom),weighted_normal_fd=float(np.dot(f,fd)/denom),weighted_separation=float(np.dot(f,sep)/denom),
      normal_force_endpoint_pairing=float(np.dot(f,vn)),normal_force_material_fd_pairing=float(np.dot(f,fd)),friction_force_endpoint_pairing=float(np.sum(fv*pv)),friction_force_material_fd_pairing=float(np.sum(fv*pfd)),
      friction_resultant_norm=float(np.linalg.norm(fv.sum(0))),friction_norm_sum=float(norms.sum()),friction_norm_sum_over_normal=float(norms.sum()/denom),
      friction_endpoint_positive_terms=int((np.sum(fv*pv,axis=1)>0).sum()),friction_endpoint_negative_terms=int((np.sum(fv*pv,axis=1)<0).sum()))
    rows.append(row)
 arrays={key:np.array([r[key] for r in rows]) for key in rows[0]}; arrays['normal_singular_values']=np.array(singular)
 assert len(rows)==48000 and normal_points==len(a['case'])
 np.savez_compressed(O/'arrays.npz',**arrays)
 summaries=[]; metrics=[k for k in rows[0] if k not in ['substep','case','side','observed_floor_only']]
 for c in range(24):
  for side in range(2):
   for lo,hi in [(1601,2199),(2200,2600)]:
    m=(arrays['case']==c)&(arrays['side']==side)&(arrays['substep']>=lo)&(arrays['substep']<=hi); total=float(arrays['normal_force'][m].sum())
    summaries.append(dict(case=c,side=['left','right'][side],window=[lo*.005,hi*.005],samples=int(m.sum()),observed_floor_only=int(arrays['observed_floor_only'][m].sum()),external_set_incomplete=int((1-arrays['observed_floor_only'][m]).sum()),metrics={key:stats(arrays[key][m]) for key in metrics},force_weighted_positive_velocity_fraction=float(arrays['positive_velocity_force'][m].sum()/total) if total else None,force_weighted_negative_velocity_fraction=float(arrays['negative_velocity_force'][m].sum()/total) if total else None,force_weighted_positive_gap_fraction=float(arrays['positive_gap_force'][m].sum()/total) if total else None))
 sources=[Path(__file__).resolve(),D/'finite_contact_revision_322.json',D/'finite_contact_source_321/SHA256.json',D/'contact_velocity_diagnosis_267/contact_points.npz',D/'contact_velocity_audit_268/independent_audit.json',D/'ankle_impulse_296/arrays.npz',P/'substep_states.npz',P/'contact_returns.npz',P/'contact_recorder_metadata.json',P/'static_inertial_model.json',P/'actuator_metadata.json']
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=scope,foot_intervals=len(rows),normal_point_records=normal_points,friction_point_records=friction_points,rows=summaries,physics_ticks=0,physical_admission=False,repair_complete=False,forward_model_identified=False,source_sha256={str(p):sha(p) for p in sources})
 (O/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)); (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2))
 print(json.dumps(dict(foot_intervals=len(rows),normal_points=normal_points,friction_points=friction_points,selected=[{k:v for k,v in r.items() if k!='metrics'} for r in summaries if r['case']<2 and r['window'][0]==11.])),flush=True)
if __name__=='__main__':main()
