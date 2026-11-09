"""Offline sole-centred correction screen on every saved234/273 raw sample.

This is a local kinematic requirement calculation, NOT a velocity predictor,
new feedback controller, force allocation or trajectory. No simulation runs.
"""
from pathlib import Path
import datetime, hashlib, json
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel, level_rotation, apply

D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
O=D/'coordinated_stance_diagnosis_286'
SIDES=['left','right']; JOINTS=['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def skew(v):
 x,y,z=v;return np.array([[0,-z,y],[z,0,-x],[-y,x,0.]])
def stats(x):
 x=np.asarray(x,float);return dict(min=float(x.min()),median=float(np.median(x)),max=float(x.max()))

def main():
 O.mkdir(exist_ok=False)
 m=BodyModel(D/'cooked_model_50/model.json');joints={j['name']:j for j in m.joints}
 limits={n:np.deg2rad(joints[n]['limits']) for n in m.names}
 ids=np.array([[m.names.index(s+'_'+j+'_joint') for j in JOINTS] for s in SIDES])
 sources=[Path(__file__).resolve(),D/'cooked_model_50/model.json',D/'coordinated_stance_revision_286.json']
 arrays={k:[] for k in ['q','qd','gravity','gyro','command','J','omega','delta','ankle_delta','corners','corner_delta','ankle_corner_delta','rank','residual']}
 rows=[]
 for run in [234,273]:
  p=D/({234:'centered_probe_234',273:'roll_feedback_probe_273'}[run])/'candidate'
  raw=read(p/'touchdown_raw_176.json');bal={(r['case'],r['tick']):r for r in read(p/'balance_feedback_161.json')};cmd=np.load(p/'fixed_ticks_132.npz')['before']
  sources.extend(p/f for f in ['touchdown_raw_176.json','balance_feedback_161.json','fixed_ticks_132.npz','results.json'])
  assert len(raw)==1900 and read(p/'results.json')['joint_names']==m.names
  for r in raw:
   c,k=r['case'],r['tick'];b=bal[c,k];q=np.array(r['q'],float);qd=np.array(r['qd'],float);g=np.array(r['gravity'],float);gyro=np.array(r['gyro'],float)
   assert np.array_equal(q,np.array(b['q'])) and np.array_equal(g,np.array(b['gravity'])) and np.array_equal(gyro,np.array(b['gyro']))
   R=level_rotation(g);fk=m.fk(q);base_omega=R@gyro
   eligible=bool(b['feedback']['at_hover'] and b['feedback']['valid'] and b['feedback']['geometry_fresh'] and b['feedback']['estimated_foot_load_fraction']>=.8)
   for si,s in enumerate(SIDES):
    foot=s+'_ankle_roll_link';vertices=m.vertices[foot];bottom=vertices[np.abs(vertices[:,2]-vertices[:,2].min())<1e-10];assert len(bottom)==4
    centre=bottom.mean(0);fc=R@apply(fk[foot],centre);points=apply(fk[foot],bottom)@R.T;J=[]
    for ji in ids[si]:
     name=m.names[ji];j=joints[name];frame=fk[j['body0'][0]]@m.frames[name][0];origin=R@frame[:3,3];axis=R@frame[:3,'XYZ'.index(j['axis'])]
     J.append(np.r_[np.cross(axis,fc-origin),axis])
    J=np.array(J).T;omega=base_omega+J[3:]@qd[ids[si]]
    # One fixed dimensional scale, borrowed from273; not tuned or a dynamic model.
    goal=np.array([0.,0.,0.,-.025*omega[0],0.,0.]);delta=np.linalg.solve(J,goal)
    ankle=np.zeros(6);ankle[5]=-.025*qd[ids[si,5]]
    proposed=q.copy();proposed[ids[si]]+=delta;fkp=m.fk(proposed)
    direct=q.copy();direct[ids[si]]+=ankle;fka=m.fk(direct)
    dp=apply(fkp[foot],bottom)@R.T-points;da=apply(fka[foot],bottom)@R.T-points
    # Full material-point constraint, pelvis fixed: preserves both translations
    # and rotations. Report rank; do not infer a dynamic-control impossibility.
    C=np.concatenate([J[:3]-skew(pt-fc)@J[3:] for pt in points]);rank=np.linalg.matrix_rank(C)
    margin=min(min(cmd[k,c,j]+delta[ii]-limits[m.names[j]][0],limits[m.names[j]][1]-cmd[k,c,j]-delta[ii]) for ii,j in enumerate(ids[si]))
    orient=R@Rotation.from_matrix(fkp[foot][:3,:3]@fk[foot][:3,:3].T).as_rotvec()
    row=dict(run=run,case=c,tick=k,t=k*.02,side=s,eligible_hover=eligible,raw_max_moving_qd=float(abs(qd[:23]).max()),measured_foot_omega_level=omega.tolist(),required_delta_rad=delta.tolist(),old_ankle_only_delta_rad=ankle.tolist(),sole_center_shift_m=dp.mean(0).tolist(),corner_vertical_displacement_m=dp[:,2].tolist(),max_corner_displacement_m=float(np.linalg.norm(dp,axis=1).max()),foot_rotation_delta_rad=orient.tolist(),joint_command_margin_rad=float(margin),per_leg_l1_delta_rad=float(abs(delta).sum()),nonankle_max_delta_rad=float(abs(delta[:4]).max()),pitch_priority_delta_rad=float(delta[4]),full_sole_constraint_rank=int(rank),linear_task_residual=float(abs(J@delta-goal).max()),full_sole_motion_norm=float(np.linalg.norm(C@delta)),applied=False)
    rows.append(row)
    for key,value in dict(q=q,qd=qd,gravity=g,gyro=gyro,command=cmd[k,c],J=J,omega=omega,delta=delta,ankle_delta=ankle,corners=points,corner_delta=dp,ankle_corner_delta=da,rank=rank,residual=J@delta-goal).items():arrays[key].append(value)
  print('completed saved run',run,'records',len(rows),flush=True)
 summaries=[]
 for run in [234,273]:
  for c in [0,1]:
   for s in SIDES:
    for lo,hi in [(1,8.98),(9,10.98),(11,14.98),(15,19.98)]:
     rr=[r for r in rows if r['run']==run and r['case']==c and r['side']==s and lo-1e-9<=r['t']<=hi+1e-9]
     summaries.append(dict(run=run,case=c,side=s,window=[lo,hi],samples=len(rr),eligible_hover=sum(r['eligible_hover'] for r in rr),raw_speed=stats([r['raw_max_moving_qd'] for r in rr]),foot_omega_x=stats([r['measured_foot_omega_level'][0] for r in rr]),required_joint_delta_median=np.median([r['required_delta_rad'] for r in rr],axis=0).tolist(),corner_z_min=stats([min(r['corner_vertical_displacement_m']) for r in rr]),corner_z_max=stats([max(r['corner_vertical_displacement_m']) for r in rr]),l1_delta=stats([r['per_leg_l1_delta_rad'] for r in rr]),pitch_priority_abs=stats([abs(r['pitch_priority_delta_rad']) for r in rr]),command_margin=stats([r['joint_command_margin_rad'] for r in rr])))
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,rows=rows,summaries=summaries,joint_names=m.names,leg_joint_order=JOINTS,leg_ids=ids.tolist(),records=len(rows),raw_samples=3800,source_sha256={str(p):sha(p) for p in sources},physics_ticks=0,optimizer_updates=0,runtime_installed=False,physical_admission=False,repair_complete=False,limitations=['All two selected cases and every saved20ms sample from each234/273 included; other22 cases have no equivalent full raw stream and are not synthesized.','Angular velocity is computed from current encoder velocity and IMU gyro; no body/contact truth or approximate torque enters the correction. Base linear velocity and common slip are unobservable here.','The solve holds the pelvis and sole centre fixed, but allows sole rotation. It is a local position correction scaled by.025s, NOT a predicted reduction in actual raw velocity.','Four material corners must move for nonzero sole rotation. Full-rank fixed-corner constraints have zero leg nullspace only in this fixed-pelvis model, not an impossibility theorem for dynamics or alternate contacts.','Required hip/knee/pitch adjustments are reported, NOT granted new authority. Existing pitch priority, per-ankle caps,quiet guards,materials and all physical criteria remain unchanged.','No command/filter/force/support certificate or full contact trajectory is proposed by this local calculation.'])
 np.savez_compressed(O/'arrays.npz',**arrays);(O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2));print(json.dumps(dict(records=len(rows),late=[r for r in summaries if r['window']==[11,14.98]])),flush=True)
if __name__=='__main__':main()
