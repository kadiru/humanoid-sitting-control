"""Post178 measured hover bias and joint-motion diagnosis; no runtime change."""
from pathlib import Path
import datetime,json,hashlib
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'touchdown_probe_178/candidate';O=D/'hover_diagnosis_187'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);raw=read(P/'touchdown_raw_176.json');guards=read(P/'fixed_guard_records_132.json');balance=read(P/'balance_feedback_161.json');names=read(P/'results.json')['joint_names'];plans=read(P/'fixed_plans_132.json');byguard={(r['case'],r['tick']):r for r in guards};bybalance={(r['case'],r['tick']):r for r in balance};rows=[]
 for case in [0,1,6,7]:
  source=sorted([r for r in raw if r['case']==case],key=lambda r:r['tick']);plan=next(r['plan'] for r in plans if r['case']==case);ids=plan['moving_ids'];q=np.array([r['q'] for r in source]);qd=np.array([r['qd'] for r in source]);times=np.array([r['tick']*.02 for r in source]);dt=np.diff(times);fd=np.diff(q,axis=0)/dt[:,None]
  windows=[]
  for label,left,right in [('hover_before_parent',11.,14.9),('hover_after_parent',17.,19.9)]:
   ii=np.flatnonzero((times>=left)&(times<=right));assert len(ii)>100
   pose_range=np.ptp(q[ii],axis=0);rawspeed=np.abs(qd[ii]);dominant=np.argmax(rawspeed[:,ids],axis=1);counts=np.bincount(dominant,minlength=len(ids));top=sorted(range(len(ids)),key=lambda j:counts[j],reverse=True)[:4]
   pitch=np.array([source[i]['settling']['pitch_error_rad'] for i in ii]);gyro=np.array([source[i]['settling']['gyro_y_rad_s'] for i in ii]);speed=np.max(rawspeed[:,ids],axis=1)
   filters=dict(pitch_within_half_degree=int((pitch<=np.deg2rad(.5)).sum()),gyro_within_02=int((abs(gyro)<=.02).sum()),moving_joint_speed_within_05=int((speed<=.05).sum()),geometry_fresh=sum(source[i]['settling']['available'] for i in ii),all_quiet=sum(source[i]['settling']['quiet'] for i in ii),original_progress_allowed=sum(byguard[(case,source[i]['tick'])]['progress_allowed'] for i in ii))
   windows.append(dict(window=label,samples=len(ii),pitch_error_deg_min=float(np.rad2deg(pitch.min())),pitch_error_deg_max=float(np.rad2deg(pitch.max())),joint_speed_max=float(speed.max()),conditions=filters,dominant_moving_joints=[dict(joint=names[ids[j]],dominant_ticks=int(counts[j]),raw_qd_abs_median=float(np.median(rawspeed[:,ids[j]])),recorded_q_range_rad=float(pose_range[ids[j]]),finite_difference_abs_median=float(np.median(abs(fd[ii[ii>0]-1,ids[j]])))) for j in top],max_moving_pose_range_rad=float(pose_range[ids].max()),mean_feedback_offsets=np.mean([bybalance[(case,source[i]['tick'])]['feedback']['offset_after'] for i in ii],axis=0).tolist()))
  rows.append(dict(case=case,windows=windows))
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,rows=rows,interpretation='Current raw joint velocity and20ms pose differences describe different sampling quantities; a discrepancy is not a proven sensor error. Pitch bias and joint-speed guards are separate blockers. Do not relax physical criteria or relabel hover as sitting. Parent15s timing is unchanged and can perturb a still-hovering state. Next work must address actual tracking/settling and parent composition, not waive the failed guard.',physics_ticks=0,optimizer_updates=0,source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),P/'touchdown_raw_176.json',P/'fixed_guard_records_132.json',P/'balance_feedback_161.json',P/'results.json']})
 (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
