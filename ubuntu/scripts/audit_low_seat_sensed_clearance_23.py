"""Offline observability audit: could measured plane+encoders warn before housing contacts?

No control changes, simulations, training or holdout. True mesh clearance is an
independent label; it is never substituted for the measured estimate.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from plan_seat_load_targets import clipped_vertices
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OUT=D/'sensed_clearance_23'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 OUT.mkdir(exist_ok=False);model_path=Path('runs/g1_kinematics_geometry_04/model.json');m=BodyModel(model_path);rows=[];curves={};hashes={str(model_path.resolve()):sha(model_path),str(Path(__file__).resolve()):sha(__file__)}
 for label,p in [('best_normal3s',D/'normal_landing3s_06/candidate'),('rejected_joint_path',D/'joint_path_17/candidate')]:
  tr=np.load(p/'trace.npz');ev=np.load(p/'every_tick.npz');res=read(p/'results.json');gs=read(p/'geometry.json');origins=np.array(res['origins']);bn=res['body_names']
  for name in ['trace.npz','every_tick.npz','results.json','geometry.json']:hashes[str(p/name)]=sha(p/name)
  for row in res['episodes']:
   i=row['case'];g=gs[row['geometry_index']];n=np.array(g['normal']);center=np.array(g['top_center']);SR=Rotation.from_euler('y',g['pitch_deg'],degrees=True).as_matrix()
   sensed=[];true_inf=[];true_finite=[]
   for k,t in enumerate(tr['t']):
    fk=m.fk(tr['q'][k,i]);geo=tr['geometry'][k,i];s=[];a=[];f=[]
    for side in ['left','right']:
     name=side+'_hip_pitch_link';v=apply(fk[name],m.vertices[name]);s.append(float((v@geo[2:5]+geo[5]).min()) if geo[11] else np.nan)
     body=tr['body'][k,i,bn.index(name)];T=transform(Rotation.from_quat(body[3:]).as_matrix(),body[:3]-origins[i]);actual=apply(T,m.vertices[name]);a.append(float(((actual-center)@n).min()))
     local=(actual-center)@SR;local[:,0]-=.4;vv=clipped_vertices(local)
     if len(vv):f.append(float(vv[:,2].min()))
    sensed.append(min(s));true_inf.append(min(a));true_finite.append(min(f) if f else np.nan)
   sensed=np.array(sensed);true_inf=np.array(true_inf);true_finite=np.array(true_finite);tt=tr['t'];valid=np.isfinite(sensed)&(tr['geometry'][:,i,11]>0);window=(tt>=8)&(tt<=15)&valid
   ids=np.flatnonzero(ev['wrong'][:,i]>10);first=float(ev['t'][ids[0]]) if len(ids) else None
   warn=np.flatnonzero(window&(sensed<.005));warn_t=float(tt[warn[0]]) if len(warn) else None
   end=first if first is not None else 15.;near=window&(tt<=end)&(abs(true_inf)<.04)
   before=np.flatnonzero(valid&(tt<=first))[-1] if first else None
   rows.append(dict(controller=label,case=i,height_m=g['height'],slope_deg=g['pitch_deg'],stable=row['success'],clean=bool(row['success'] and not any(row['failures'].values()) and row['slip_ticks']==0 and row['illegal_ticks']==0),slip_ticks=row['slip_ticks'],first_forbidden_s=first,
    first_sensed_clearance_below_5mm_s=warn_t,warning_lead_s=first-warn_t if first is not None and warn_t is not None else None,
    last_sample_before_contact=dict(t=float(tt[before]),sensed_m=float(sensed[before]),true_unclipped_m=float(true_inf[before]),true_finite_patch_m=float(true_finite[before]),age_s=float(tr['geometry'][before,i,12])) if before is not None else None,
    near_plane_samples=int(near.sum()),max_sensed_unclipped_error_near_plane_m=float(abs(sensed[near]-true_inf[near]).max()) if near.any() else None,
    minimum_sensed_m=float(sensed[window].min()) if window.any() else None,minimum_true_patch_m=float(np.nanmin(true_finite[window])) if np.isfinite(true_finite[window]).any() else None))
   curves[f'{label}_{i}']=np.stack([tt,sensed,true_inf,true_finite,tr['geometry'][:,i,12]],1)
 summary={}
 for label in ['best_normal3s','rejected_joint_path']:
  rr=[r for r in rows if r['controller']==label];contact=[r for r in rr if r['first_forbidden_s'] is not None]
  summary[label]=dict(contact_cases=len(contact),contacts_with_prior_5mm_warning=sum(r['warning_lead_s'] is not None and r['warning_lead_s']>0 for r in contact),clean_cases=sum(r['clean'] for r in rr),clean_cases_with_5mm_warning=sum(r['clean'] and r['first_sensed_clearance_below_5mm_s'] is not None for r in rr))
 np.savez_compressed(OUT/'curves.npz',**curves)
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Offline sensor-only housing clearance estimate compared with true geometry labels. The5mm line is a diagnostic warning, not a changed physical threshold or an admitted safety controller. Infinite measured plane is conservative outside the observed seat footprint; finite true patch is reported separately.0.1s sensor/pose samples bracket0.02s contact events.',simulations=0,ppo_updates=0,fresh_holdout=False,summary=summary,rows=rows,input_hashes=hashes,inputs_unchanged=all(sha(p)==h for p,h in hashes.items()))
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
