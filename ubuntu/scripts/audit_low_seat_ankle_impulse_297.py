"""Independent moving-COM impulse identity and raw-source replay for296."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'contact_wrench_diagnostic_259';O=D/'ankle_impulse_audit_297'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def rot(q,v):
 q=np.asarray(q,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True);v=np.asarray(v,float);u=q[...,:3]
 return v+2*np.cross(u,np.cross(u,v)+q[...,3,None]*v)
def median(x):
 a=sorted(map(float,x));n=len(a);return (a[(n-1)//2]+a[n//2])/2
def main():
 O.mkdir(exist_ok=False);r=read(D/'ankle_impulse_296/results.json');assert all(sha(p)==h for p,h in r['source_sha256'].items())
 s=dict(np.load(P/'substep_states.npz'));launch=dict(np.load(P/'implicit_launches.npz'));z=dict(np.load(D/'ankle_impulse_296/arrays.npz'));b=dict(np.load(D/'momentum_balance_294/arrays.npz'));inert=read(D/'contact_dynamics_screen_v3_290/inertial_model.json');geom=read(D/'cooked_model_50/model.json');meta=read(P/'actuator_metadata.json');X=s['link_transforms'].astype(float);V=s['link_velocities'].astype(float);errors={}
 def check(key,x,y,tol=1e-10):
  x=np.asarray(x);y=np.asarray(y);assert x.shape==y.shape,(key,x.shape,y.shape);err=float(np.max(abs(x.astype(float)-y.astype(float))));errors[key]=err;assert err<=tol,(key,err,tol)
 check('coverage',z['coverage'],b['coverage'],0);check('time',z['t'],s['t'][1:],0)
 wr={(a['substep'],a['side']):a for a in read(D/'contact_wrench_audit_260/contact_wrenches.json')};summary={(a['case'],a['side'],tuple(a['window'])):a for a in r['rows']};statistics=0
 for side in ['left','right']:
  joint=next(j for j in geom['joints'] if j['name']==side+'_ankle_roll_joint');child=meta['body_names'].index(joint['body1'][0]);parent=meta['body_names'].index(joint['body0'][0]);ji=meta['joint_names'].index(joint['name']);mass=inert['mass'][child];Ib=np.array(inert['inertia_body'][child]);q=X[:,:,child,3:];qp=X[:,:,parent,3:]
  com=X[:,:,child,:3]+rot(q,np.broadcast_to(inert['local_com'][child],q.shape[:-1]+(3,)));momentum=mass*V[:,:,child,:3];conj=q.copy();conj[...,:3]*=-1;spin=rot(q,rot(conj,V[:,:,child,3:])@Ib.T)
  pos=X[:,:,parent,:3]+rot(qp,np.broadcast_to(joint['p0'],qp.shape[:-1]+(3,)));v=np.eye(3)['XYZ'.index(joint['axis'])];localaxis=rot(joint['q0'],v);axis=rot(qp,np.broadcast_to(localaxis,qp.shape[:-1]+(3,)));axmid=axis[1:]+axis[:-1];axmid/=np.linalg.norm(axmid,axis=-1,keepdims=True)
  pmid=(momentum[1:]+momentum[:-1])/2;cmid=(com[1:]+com[:-1])/2;jmid=(pos[1:]+pos[:-1])/2;dp=np.diff(momentum,axis=0);dc=np.diff(com,axis=0);ds=np.diff(spin,axis=0)
  F=np.array([wr[k,side]['force_world_N'] for k in range(1601,2601)]);M=np.array([wr[k,side]['moment_about_link_world_Nm'] for k in range(1601,2601)]);gforce=np.array([0.,0.,-mass*9.81])
  # Discrete product-rule identity around the moving COM, independent of
  # screen296's fixed-origin orbital momentum and boundary-force subtraction.
  required=(ds+np.cross(cmid-jmid,dp)+np.cross(dc,pmid))/.005
  required-=M+np.cross(X[1:,:,child,:3]-jmid,F)+np.cross(cmid-jmid,gforce)
  axial=(axmid*required).sum(-1);boundaryF=dp/.005-F-gforce;telemetry=launch['applied_effort'][:,:,ji].astype(float);armterm=meta['armature'][0][ji]*np.diff(s['qd'][:,:,ji].astype(float),axis=0)/.005
  for key,x in dict(com=com,momentum=momentum,spin=spin,joint=pos,axis=axis,axis_mid=axmid,boundary_force=boundaryF,boundary_moment=required,axial=axial,telemetry=telemetry,armature_acceleration_term=armterm,contact_force=F,contact_moment=M).items():check(side+'_'+key,x,z[side+'_'+key])
  for c in range(24):
   for low,high in [(8.,11.),(11.,13.)]:
    inds=[k for k in range(1000) if s['t'][k]>=low-1e-12 and s['t'][k+1]<=high+1e-12];ok=[k for k in inds if b['coverage'][k,c]];row=summary[c,side,(low,high)];assert len(ok)==row['available_intervals'] and len(inds)==row['intervals']
    for name,v in [('required_boundary_Nm',axial),('approximate_PD_Nm',telemetry),('telemetry_minus_boundary_Nm',telemetry-axial),('armature_acceleration_Nm',armterm)]:
     if not ok:assert row[name] is None;continue
     data=v[ok,c];stats=dict(median=median(data),RMS=(sum(float(x)**2 for x in data)/len(data))**.5,maximum_absolute=max(abs(float(x)) for x in data))
     for field,value in stats.items():check(f'stat{statistics}_{field}',value,row[name][field]);statistics+=1
 assert all(sha(p)==h for p,h in r['source_sha256'].items())
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,foot_case_intervals=48000,statistics=statistics,maximum_error=max(errors.values()),errors=errors,arithmetic_tolerance=1e-10,physics_ticks=0,optimizer_updates=0,physical_admission=False,repair_complete=False,limitations=r['limitations'],source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),D/'ankle_impulse_296/results.json',D/'ankle_impulse_296/arrays.npz',D/'ankle_impulse_revision_296.json']})
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2,allow_nan=False));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2));print(json.dumps({k:out[k] for k in ['passed','foot_case_intervals','statistics','maximum_error','physical_admission']}),flush=True)
if __name__=='__main__':main()
