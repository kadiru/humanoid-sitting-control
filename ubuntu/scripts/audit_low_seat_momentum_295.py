"""Independent body-momentum and fixed-window impulse replay for294."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'contact_wrench_diagnostic_259';O=D/'momentum_audit_295'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def rotate(q,v):
 q=q/np.linalg.norm(q,axis=-1,keepdims=True);u=q[...,:3];w=q[...,3,None]
 return v+2*np.cross(u,np.cross(u,v)+w*v)
def main():
 O.mkdir(exist_ok=False);r=read(D/'momentum_balance_294/results.json');s=dict(np.load(P/'substep_states.npz'));e=dict(np.load(P/'every_tick.npz'));z=dict(np.load(D/'momentum_balance_294/arrays.npz'));model=read(D/'contact_dynamics_screen_v3_290/inertial_model.json');names=model['body_names']
 assert all(sha(p)==h for p,h in r['source_sha256'].items());errors={}
 def check(k,a,b,tol=1e-10):
  a=np.asarray(a);b=np.asarray(b);assert a.shape==b.shape,(k,a.shape,b.shape)
  err=float(np.max(abs(a.astype(float)-b.astype(float))));errors[k]=err;assert err<=tol,(k,err,tol)
 X=s['link_transforms'].astype(float);V=s['link_velocities'].astype(float);m=np.array(model['mass']);offset=np.array(model['local_com']);Ib=np.array(model['inertia_body']);n=len(X)
 origin=X[0,:,names.index('pelvis'),:3];pt=np.zeros((n,24,3));spin=np.zeros_like(pt);weighted_com=np.zeros_like(pt);orbital=np.zeros_like(pt);g_moment=np.zeros_like(pt)
 # Independent quaternion-vector rotations and bodywise assembly. H uses
 # each body's world orbital momentum first, then shifts to fixed O.
 for b in range(44):
  q=X[:,:,b,3:];com=X[:,:,b,:3]+rotate(q,np.broadcast_to(offset[b],(n,24,3)));p=m[b]*V[:,:,b,:3];pt+=p;weighted_com+=m[b]*com
  conj=q.copy();conj[...,:3]*=-1;wl=rotate(conj,V[:,:,b,3:]);sl=wl@Ib[b].T;spin+=rotate(q,sl);orbital+=np.cross(com,p)
 H=spin+orbital-np.cross(origin[None],pt)
 g=np.array([0.,0.,-9.81]);g_moment=np.cross(weighted_com-m.sum()*origin[None],g)
 check('P',pt,z['P']);check('H',H,z['H']);check('origin',origin,z['origin'],0);check('gravity_moment',g_moment,z['gravity_moment'])
 F=np.zeros_like(pt);T=np.zeros_like(pt)
 for wr in read(D/'contact_wrench_audit_260/contact_wrenches.json'):
  k=wr['substep']-1600;b=names.index(wr['side']+'_ankle_roll_link');force=np.asarray(wr['force_world_N']);moment=np.asarray(wr['moment_about_link_world_Nm']);F[k]+=force;T[k]+=moment+np.cross(X[k,:,b,:3]-origin,force)
 check('foot_force',F,z['foot_force']);check('foot_moment',T,z['foot_moment'])
 dF=np.empty((1000,24,3));dT=np.empty_like(dF)
 for k in range(1000):dF[k]=(F[k+1]+m.sum()*g)*.005;dT[k]=(T[k+1]+(g_moment[k]+g_moment[k+1])/2)*.005
 check('impulse_force',dF,z['impulse_force']);check('impulse_moment',dT,z['impulse_moment'])
 check('gravity_sampling_difference',.0025*np.diff(g_moment,axis=0),z['gravity_sampling_difference'])
 contact_names=read(P/'contact_body_names.json');bad=[j for j,b in enumerate(contact_names) if b not in ['left_ankle_roll_link','right_ankle_roll_link']];available=[]
 for sub in range(1601,2601):
  tick=(sub-1)//4;available.append(np.all(e['unexpected_body_normal'][tick][:,bad]==0,axis=1)&(e['wrong'][tick]==0))
 available=np.array(available);check('coverage',available,z['coverage'],0)
 assert int(available.sum())==r['observed_floor_only_case_intervals']
 cf=np.concatenate([np.zeros((1,24,3)),np.cumsum(dF,axis=0)]);ct=np.concatenate([np.zeros((1,24,3)),np.cumsum(dT,axis=0)])
 summaries={(a['case'],tuple(a['window']),a['interval_ms']):a for a in r['summaries']};stats=0
 for width in [1,4,20,200]:
  start=np.arange(0,1000,width);end=start+width;dt=width*.005;dp=pt[end]-pt[start];dh=H[end]-H[start];fi=cf[end]-cf[start];mi=ct[end]-ct[start];rf=(dp-fi)/dt;rt=(dh-mi)/dt
  av=np.stack([np.all(available[x:y],axis=0) for x,y in zip(start,end)])
  for key,v in dict(starts=start,ends=end,delta_P=dp,delta_H=dh,impulse_F=fi,impulse_T=mi,residual_F=rf,residual_T=rt,available=av).items():check(f'w{width}_{key}',v,z[f'w{width}_{key}'])
  # Translation covariance is checked, not used to fit the origin.
  shift=np.array([.137,-.211,.083]);hs=H-np.cross(shift,pt);tis=dT-np.cross(shift,dF)
  check(f'w{width}_origin_covariance',((hs[end]-hs[start])-np.stack([tis[x:y].sum(0) for x,y in zip(start,end)]))/dt,rt-np.cross(shift,rf))
  for c in range(24):
   for low,high in [(8.,11.),(11.,13.)]:
    selected=[j for j,(x,y) in enumerate(zip(start,end)) if s['t'][x]>=low-1e-12 and s['t'][y]<=high+1e-12];ok=[j for j in selected if av[j,c]];row=summaries[c,(low,high),dt*1000]
    assert len(selected)==row['intervals'] and len(ok)==row['available_intervals']
    if not ok:assert row['force_residual_RMS_N'] is None and row['moment_residual_RMS_Nm'] is None;continue
    for key,data in [('force',rf),('moment',rt)]:
     unit='N' if key=='force' else 'Nm';v=data[ok,c];mean=np.array([sum(float(x) for x in v[:,axis])/len(v) for axis in range(3)]);rms=(sum(sum(float(x)**2 for x in vv) for vv in v)/len(v))**.5
     check(f'stat{stats}_mean',mean,row[f'{key}_residual_mean_{unit}']);check(f'stat{stats}_RMS',rms,row[f'{key}_residual_RMS_{unit}']);stats+=1
 assert all(sha(p)==h for p,h in r['source_sha256'].items())
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,arithmetic_tolerance=1e-10,case_state_records=24024,body_state_records=1057056,case_intervals=24000,summary_statistics=stats,maximum_error=max(errors.values()),errors=errors,physical_admission=False,repair_complete=False,physics_ticks=0,optimizer_updates=0,limitations=r['limitations'],source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),D/'momentum_balance_294/arrays.npz',D/'momentum_balance_294/results.json',D/'momentum_revision_294.json']})
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2,allow_nan=False));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2));print(json.dumps({k:out[k] for k in ['passed','case_intervals','summary_statistics','maximum_error','physical_admission']}),flush=True)
if __name__=='__main__':main()
