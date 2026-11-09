"""Recorded terminal-foot boundary impulse, not a motor-torque sensor or controller."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial.transform import Rotation
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'contact_wrench_diagnostic_259';O=D/'ankle_impulse_296'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert read(D/'momentum_audit_295/independent_audit.json')['passed'];assert not O.exists()
 files=[P/'substep_states.npz',P/'actuator_metadata.json',P/'implicit_launches.npz',D/'cooked_model_50/model.json',D/'contact_dynamics_screen_v3_290/inertial_model.json',D/'contact_wrench_audit_260/contact_wrenches.json',D/'momentum_balance_294/arrays.npz',D/'momentum_audit_295/independent_audit.json',Path(__file__).resolve()]
 hashes={str(p):sha(p) for p in files};scope=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,source_sha256=hashes,physics_ticks=0,optimizer_updates=0,alignment='Post-step contact interval(k-1,k], midpoint joint position/normalized mean axis. No fitted delay, torque offset, inertia or gain.',interpretation='Required equivalent ankle-roll boundary axial moment under a finite-interval rigid-body momentum model. Includes unresolved drive/constraint/armature effects; not directly measured actuator torque. No feedback, effort replacement or physical admission.')
 assert not (D/'ankle_impulse_revision_296.json').exists();(D/'ankle_impulse_revision_296.json').write_text(json.dumps(scope,indent=2));O.mkdir()
 s=dict(np.load(P/'substep_states.npz'));launch=dict(np.load(P/'implicit_launches.npz'));meta=read(P/'actuator_metadata.json');geom=read(D/'cooked_model_50/model.json');inert=read(D/'contact_dynamics_screen_v3_290/inertial_model.json');balance=dict(np.load(D/'momentum_balance_294/arrays.npz'));wrenches={(r['substep'],r['side']):r for r in read(D/'contact_wrench_audit_260/contact_wrenches.json')}
 X=s['link_transforms'].astype(float);V=s['link_velocities'].astype(float);bn=meta['body_names'];jn=meta['joint_names'];jmap={j['name']:j for j in geom['joints']};Oref=balance['origin'];dt=.005;g=np.array([0.,0.,-9.81]);arr=dict(t=s['t'][1:],coverage=balance['coverage']);rows=[]
 for side in ['left','right']:
  name=side+'_ankle_roll_joint';j=jmap[name];child=j['body1'][0];parent=j['body0'][0];assert not any(x['body0'][0]==child for x in geom['joints']);b=bn.index(child);a=bn.index(parent);ji=jn.index(name);mass=inert['mass'][b]
  R=Rotation.from_quat(X[:,:,b,3:].reshape(-1,4)).as_matrix().reshape(1001,24,3,3);Rp=Rotation.from_quat(X[:,:,a,3:].reshape(-1,4)).as_matrix().reshape(1001,24,3,3)
  com=X[:,:,b,:3]+np.einsum('tcij,j->tci',R,inert['local_com'][b]);momentum=mass*V[:,:,b,:3];Iw=np.einsum('tcij,jk,tclk->tcil',R,inert['inertia_body'][b],R);spin=np.einsum('tcij,tcj->tci',Iw,V[:,:,b,3:]);H=spin+np.cross(com-Oref[None],momentum)
  joint=X[:,:,a,:3]+np.einsum('tcij,j->tci',Rp,j['p0']);axis=np.einsum('tcij,j->tci',Rp,Rotation.from_quat(j['q0']).as_matrix()[:,'XYZ'.index(j['axis'])]);jointmid=(joint[1:]+joint[:-1])/2;axismid=axis[1:]+axis[:-1];axismid/=np.linalg.norm(axismid,axis=-1,keepdims=True)
  force=np.array([wrenches[k,side]['force_world_N'] for k in range(1601,2601)]);moment=np.array([wrenches[k,side]['moment_about_link_world_Nm'] for k in range(1601,2601)])
  com_mid=(com[1:]+com[:-1])/2;extmoment=moment+np.cross(X[1:,:,b,:3]-Oref[None],force)+np.cross(com_mid-Oref[None],mass*g)
  boundaryF=np.diff(momentum,axis=0)/dt-force-mass*g
  boundaryMoment_O=np.diff(H,axis=0)/dt-extmoment;boundaryMoment_joint=boundaryMoment_O-np.cross(jointmid-Oref[None],boundaryF);axial=np.sum(axismid*boundaryMoment_joint,axis=-1)
  # The exact recorded prelaunch telemetry is compared separately. It is not
  # used to compute momentum or fitted to the required boundary moment.
  telemetry=launch['applied_effort'][:,:,ji].astype(float);assert np.array_equal(launch['substep_before'],np.arange(1600,2600))
  # Armature acceleration is a descriptive missing-term sensitivity, not an
  # asserted physical rotor model or a correction to reported boundary moments.
  arm=meta['armature'][0][ji];armterm=arm*np.diff(s['qd'][:,:,ji].astype(float),axis=0)/dt
  for k,v in dict(com=com,momentum=momentum,spin=spin,joint=joint,axis=axis,axis_mid=axismid,boundary_force=boundaryF,boundary_moment=boundaryMoment_joint,axial=axial,telemetry=telemetry,armature_acceleration_term=armterm,contact_force=force,contact_moment=moment).items():arr[side+'_'+k]=v
  for c in range(24):
   for low,high in [(8.,11.),(11.,13.)]:
    use=(s['t'][:-1]>=low-1e-12)&(s['t'][1:]<=high+1e-12);ok=use&balance['coverage'][:,c]
    r=dict(case=c,side=side,window=[low,high],intervals=int(use.sum()),available_intervals=int(ok.sum()),incomplete_intervals=int((use&~balance['coverage'][:,c]).sum()))
    for k,v in [('required_boundary_Nm',axial),('approximate_PD_Nm',telemetry),('telemetry_minus_boundary_Nm',telemetry-axial),('armature_acceleration_Nm',armterm)]:r[k]=dict(median=float(np.median(v[ok,c])),RMS=float(np.sqrt(np.mean(v[ok,c]**2))),maximum_absolute=float(abs(v[ok,c]).max())) if ok.any() else None
    rows.append(r)
 limits=['Terminal feet have no model descendants; a single-body impulse balance is used. Joint origin/axis come from recorded parent poses and unchanged model frames.', 'Finite5ms impulses are represented by an equivalent force/couple at the midpoint joint origin with normalized mean axis. Solver-internal application times and moving-axis impulses are unobserved.', 'Authored inertia, frozen COM/mass and configured gravity retain294 limitations; no effective-live rotor/spatial armature model is available. Armature times raw-velocity difference is only a separately reported scale.', 'Only intervals with zero enclosing non-foot normal history and wrong resultants are summarized; this does not rule out cancelling/unrecorded external forces.', 'Approximate PD telemetry is never used in the boundary inverse calculation and is not direct solver/hardware effort. A difference cannot uniquely identify drive, constraint, bias, sampling or inertia cause.', 'No controller, damping/gain, solver flag, quiet guard, physical criterion or reported effort result is changed. No physical trial/PPO/holdout/video motion is added.']
 assert all(sha(p)==h for p,h in hashes.items());np.savez_compressed(O/'arrays.npz',**arr);out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=scope,rows=rows,case_intervals_per_foot=24000,foot_case_intervals=48000,limitations=limits,source_sha256=hashes,physics_ticks=0,optimizer_updates=0,physical_admission=False,repair_complete=False);(O/'results.json').write_text(json.dumps(out,indent=2,allow_nan=False));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2));print(json.dumps([r for r in rows if r['case'] in [0,1] and r['window']==[11.,13.]]),flush=True)
if __name__=='__main__':main()
