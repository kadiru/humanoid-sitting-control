"""Independent finite-difference FK and constrained-solve audit of286v2."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,level_rotation,apply
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'coordinated_stance_diagnosis_v2_286';O=D/'coordinated_stance_audit_v2_287'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);r=read(P/'results.json');a=np.load(P/'arrays.npz');assert len(r['rows'])==7600
 assert all(sha(p)==h for p,h in r['source_sha256'].items());m=BodyModel(D/'cooked_model_50/model.json');ids=r['leg_ids'];eps=1e-6
 sources={}
 for run,n in [(234,'centered_probe_234'),(273,'roll_feedback_probe_273')]:
  p=D/n/'candidate';sources[run]=( {(x['case'],x['tick']):x for x in read(p/'touchdown_raw_176.json')},dict(np.load(p/'fixed_ticks_132.npz')) )
 maxima=np.zeros(6);errors=[];ranks=[];crosschecks=0;terminal=0
 for i,row in enumerate(r['rows']):
  run,c,k=row['run'],row['case'],row['tick'];raw,ticks=sources[run];src=raw[c,k]
  for key in ['q','qd','gravity','gyro']:assert np.array_equal(a[key][i],src[key])
  command=ticks['before'][k,c] if k<1000 else ticks['after'][-1,c];assert np.array_equal(command,a['command'][i]);assert row['subsequent_applied_interval_available']==(k<1000);terminal+=int(k==1000);crosschecks+=1
  si=['left','right'].index(row['side']);leg=ids[si];foot=row['side']+'_ankle_roll_link';q=a['q'][i];R=level_rotation(a['gravity'][i]);v=m.vertices[foot];bottom=v[np.abs(v[:,2]-v[:,2].min())<1e-10];centre=np.mean(bottom,axis=0);J=[];C=[]
  for j in leg:
   plus=q.copy();minus=q.copy();plus[j]+=eps;minus[j]-=eps;fp=m.fk(plus)[foot];fm=m.fk(minus)[foot]
   linear=R@(apply(fp,centre)-apply(fm,centre))/(2*eps);angular=R@Rotation.from_matrix(fp[:3,:3]@fm[:3,:3].T).as_rotvec()/(2*eps)
   J.append(np.r_[linear,angular]);C.append(((apply(fp,bottom)-apply(fm,bottom))@R.T/(2*eps)).ravel())
  J=np.array(J).T;C=np.array(C).T;rank=np.linalg.matrix_rank(C);assert rank==row['full_sole_constraint_rank']==6;ranks.append(rank)
  omega=R@a['gyro'][i]+J[3:]@a['qd'][i,leg];goal=np.r_[np.zeros(3),-.025*a['omega'][i,0],0.,0.]
  # Independent least squares on exact stored map; finite differences separately
  # verify that map, avoiding differencing noise in the solve arithmetic check.
  delta=np.linalg.lstsq(a['J'][i],goal,rcond=None)[0]
  qp=q.copy();qp[leg]+=a['delta'][i];p0=apply(m.fk(q)[foot],bottom)@R.T;pd=apply(m.fk(qp)[foot],bottom)@R.T-p0
  vals=np.array([abs(J-a['J'][i]).max(),abs(omega-a['omega'][i]).max(),abs(delta-a['delta'][i]).max(),abs(pd-a['corner_delta'][i]).max(),abs(np.linalg.norm(C@a['delta'][i])-row['full_sole_motion_norm']),abs(a['J'][i]@a['delta'][i]-goal).max()])
  assert vals[0]<2e-8 and vals[1]<2e-8 and vals[2]<1e-10 and vals[3]==0 and vals[4]<2e-9 and vals[5]<1e-12,(i,vals)
  maxima=np.maximum(maxima,vals);errors.append(vals)
  if i%1000==0:print('audited',i,flush=True)
 assert terminal==8 and crosschecks==7600
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,records=crosschecks,terminal_records_with_no_next_interval=terminal,finite_difference_step_rad=eps,maximum_errors=dict(zip(['Jacobian','angular_velocity','least_squares_delta','finite_corner_displacement','full_corner_constraint','task_residual'],maxima.tolist())),full_sole_rank_min=int(min(ranks)),full_sole_rank_max=int(max(ranks)),physics_ticks=0,optimizer_updates=0,physical_admission=False,repair_complete=False,interpretation='All saved inputs and command clocks checked. Kinematic and linear-algebra arithmetic only; no predicted closed-loop velocity reduction, new authority, force/clearance/physical admission or impossibility proof.',source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),P/'results.json',P/'arrays.npz',D/'cooked_model_50/model.json']})
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));np.savez_compressed(O/'errors.npz',errors=errors);(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
