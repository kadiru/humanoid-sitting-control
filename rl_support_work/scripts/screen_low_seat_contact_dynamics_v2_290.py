"""New offline inertial/drive sensitivity screen; no physical admission."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd,UsdPhysics,UsdGeom
from low_seat_contact_dynamics_290 import ContactDynamics
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'contact_dynamics_screen_v2_290'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False)
 asset=D/'asset_contact_47/g1.usd';base=D/'drive_diagnostic_246/static_inertial_model.json';actpath=D/'drive_diagnostic_246/actuator_metadata.json';geo=D/'cooked_model_50/model.json'
 inputs=[asset,base,actpath,geo,Path(__file__).resolve(),Path(__file__).with_name('low_seat_contact_dynamics_290.py').resolve(),D/'contact_dynamics_revision_290.json']
 hashes={str(p):sha(p) for p in inputs};st=Usd.Stage.Open(str(asset));assert UsdGeom.GetStageMetersPerUnit(st)==UsdPhysics.GetStageKilogramsPerUnit(st)==1
 inert=read(base);inert.update(inertia_body=[],authored_properties=[],scope='Separate OFFLINE mass model: authored USD rotational tensors plus preserved backend-exported mass/COM. Not certified effective live inertia or an actual solver model.');maxcom=0.
 for i,n in enumerate(inert['body_names']):
  p=st.GetPrimAtPath('/g1/'+n);a=UsdPhysics.MassAPI(p);mass=float(a.GetMassAttr().Get());com=np.array(a.GetCenterOfMassAttr().Get(),float);diag=np.array(a.GetDiagonalInertiaAttr().Get(),float);q=a.GetPrincipalAxesAttr().Get();xyzw=np.r_[q.GetImaginary(),q.GetReal()].astype(float)
  assert mass>0 and inert['mass'][i]>0 and np.isfinite(diag).all() and (diag>0).all()
  sentinel=not np.isfinite(com).all();zeroaxes=np.linalg.norm(xyzw)==0
  if sentinel:assert n in ['pelvis_contour_link','imu_link'] and inert['local_com'][i]==[0.,0.,0.]
  else:maxcom=max(maxcom,float(abs(com-inert['local_com'][i]).max()))
  if zeroaxes:assert np.all(diag==diag[0]);Ib=np.eye(3)*diag[0]
  else:R=Rotation.from_quat(xyzw).as_matrix();Ib=R@np.diag(diag)@R.T
  assert np.linalg.eigvalsh(Ib).min()>0
  inert['inertia_body'].append(Ib.tolist());inert['authored_properties'].append(dict(body=n,mass=mass,frozen_mass=inert['mass'][i],mass_exact=(mass==inert['mass'][i]),mass_difference_kg=mass-inert['mass'][i],com=None if sentinel else com.tolist(),com_unauthored_sentinel=sentinel,principal_axes_xyzw=xyzw.tolist(),diagonal=diag.tolist(),isotropic_orientation_irrelevant=zeroaxes))
 assert maxcom<1e-6
 inert['authored_mass_exact_frozen']=all(x['mass_exact'] for x in inert['authored_properties']);inert['maximum_authored_mass_difference_kg']=max(abs(x['mass_difference_kg']) for x in inert['authored_properties']);inert['maximum_authored_vs_frozen_com_difference_m']=maxcom;inert['source_sha256']=hashes
 (O/'inertial_model.json').write_text(json.dumps(inert,indent=2,allow_nan=False))
 act=read(actpath)
 for key in ['stiffness','damping','armature']:assert np.all(np.array(act[key])==np.array(act[key])[0])
 dyn=ContactDynamics(geo,inert,act);rows=[];arrays={k:[] for k in ['q','gravity','M','C','dq','dv','free_dq','free_dv','raw_qd','gyro']}
 for run,folder in [(234,'centered_probe_234'),(273,'roll_feedback_probe_273')]:
  p=D/folder/'candidate';rawpath=p/'touchdown_raw_176.json';raw=read(rawpath);hashes[str(rawpath)]=sha(rawpath);assert len(raw)==1900
  for obs in raw:
   q=np.array(obs['q']);g=np.array(obs['gravity']);M,C,_=dyn.matrices(q,g);x,v,xf,vf,L,G=dyn.response(M,C)
   # A sensitivity derivative, not an actual next velocity or a command proposal.
   # .00075 is the existing maximum applied roll-offset change per20ms; an
   # optimistic independent box, before remaining total-authority/headroom limits.
   bound=.00075*np.abs(v[6+np.array(dyn.roll)]).sum(1)
   r=dict(run=run,case=obs['case'],tick=obs['tick'],t=obs['tick']*.02,mass_eigen_min=float(np.linalg.eigvalsh(M).min()),contact_rank=int(np.linalg.matrix_rank(C)),schur_condition=float(np.linalg.cond(G)),contact_response_residual=float(max(abs(C@v).max(),abs(C@x).max())),roll_dv_per_rad=v[6+np.array(dyn.roll)].tolist(),roll_dv_box_bound=bound.tolist(),pelvis_roll_dq_box_bound=float(.00075*abs(x[3]).sum()),pelvis_lateral_dq_box_bound=float(.00075*abs(x[1]).sum()),raw_ankle_roll_qd=np.array(obs['qd'])[dyn.roll].tolist(),raw_max_moving_qd=float(abs(np.array(obs['qd'])[:23]).max()),has_next_recorded_interval=obs['tick']<1000,applied=False)
   assert r['mass_eigen_min']>0 and r['contact_rank']==12 and r['contact_response_residual']<1e-9
   rows.append(r)
   for k,val in dict(q=q,gravity=g,M=M,C=C,dq=x,dv=v,free_dq=xf,free_dv=vf,raw_qd=obs['qd'],gyro=obs['gyro']).items():arrays[k].append(val)
  print('finished',run,len(rows),flush=True)
 summaries=[]
 for run in [234,273]:
  for c in [0,1]:
   for lo,hi in [(1,8.98),(9,10.98),(11,14.98),(15,19.98)]:
    a=[r for r in rows if r['run']==run and r['case']==c and lo<=r['t']+1e-9 and r['t']<=hi+1e-9]
    summaries.append(dict(run=run,case=c,window=[lo,hi],records=len(a),raw_roll_median=np.median([r['raw_ankle_roll_qd'] for r in a],axis=0).tolist(),modeled_roll_dv_bound_median=np.median([r['roll_dv_box_bound'] for r in a],axis=0).tolist(),pelvis_roll_dq_bound_median=float(np.median([r['pelvis_roll_dq_box_bound'] for r in a])),pelvis_lateral_dq_bound_median=float(np.median([r['pelvis_lateral_dq_box_bound'] for r in a]))))
 limitations=['Frozen linear perturbation model with two bilateral rigid foot contacts; does not model contact creation/loss, friction inequalities, solver stabilization/bias or changing mass/Jacobians.','Four backward-Euler5ms substeps for PD incremental response are an explicitly assumed offline discretization, not an implementation claim about PhysX implicit drives.','Maps are derivatives about a hypothetical contact-consistent baseline, not predictions of raw next qd. Actual observed foot velocities do not satisfy ideal sticking; do not project/substitute them into runtime.','Only existing ankle-roll command channels are varied in the derivative. Pitch/hip/finger target derivatives are zero. No new authority or command installed.','Independent .00075rad per-side box is an optimistic existing per20ms offset-rate envelope; actual authority/headroom/composition can further restrict it. It does not establish reachable quiet states.','Authored USD tensors plus preserved mass/COM and armature are not certified effective live mass matrices. Isotropic tensors for two tiny placeholder bodies make their zero principal-axis quaternion irrelevant; finite COM uses original backend export.','Six finger masses differ between authored and frozen exports by one float32 step; exact authored-mass equality is FALSE. Frozen masses remain used without adjustment; live model equality is not claimed.','No physical pass, complete contact transition, quiet/load release, torque saving, impossibility proof, PPO or holdout result follows.']
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,records=len(rows),rows=rows,summaries=summaries,limitations=limitations,roll_ids=dyn.roll,source_sha256=hashes,physics_ticks=0,initial_settling_ticks=0,optimizer_updates=0,runtime_installed=False,physical_admission=False,repair_complete=False)
 assert all(sha(p)==h for p,h in hashes.items());np.savez_compressed(O/'arrays.npz',**arrays);(O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2));print(json.dumps(dict(records=len(rows),max_com_difference=maxcom,late=[s for s in summaries if s['window']==[11,14.98]])),flush=True)
if __name__=='__main__':main()
