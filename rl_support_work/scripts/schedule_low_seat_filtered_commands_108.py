"""Construct a command-only schedule on fixed106 polyline; no simulation."""
from pathlib import Path
import datetime, hashlib, json
import numpy as np
from humanoid_sitting.shared_frame import BodyModel

D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
O=D/'contact_filtered_schedule_108'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    O.mkdir(exist_ok=False)
    source=D/'contact_load_sharing_106/results.json'
    audit=D/'contact_load_filter_audit_107/independent_audit.json'
    assert read(audit)['passed']
    data=read(source);assert data['summary']['model_command_gate']
    m=BodyModel(D/'cooked_model_50/model.json');names=m.names
    ids=[names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]
    ids += [names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']]
    limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'}
    lo=np.array([limits[names[j]][0]+.03 for j in ids],np.float32)
    hi=np.array([limits[names[j]][1]-.03 for j in ids],np.float32)
    Q=np.array([v['command'] for v in data['nodes']]);arc=np.r_[0,np.cumsum(np.max(abs(np.diff(Q,axis=0)),axis=1))]
    def at(s):
        k=min(int(np.searchsorted(arc,s,side='right')-1),len(Q)-2)
        k=max(k,0);fraction=0. if arc[k+1]==arc[k] else (s-arc[k])/(arc[k+1]-arc[k])
        return (Q[k]+fraction*(Q[k+1]-Q[k])).astype(np.float32)
    applied=[Q[0].astype(np.float32)];positions=[0.];targets=[];desired=[applied[0].copy()]
    step=np.float32(.003);alpha=np.float32(.1);s=0.;max_ticks=1500
    for _ in range(max_ticks):
        if s==arc[-1]:break
        prev=applied[-1]
        def valid(candidate):
            goal=at(candidate);raw=prev[ids]+(goal[ids]-prev[ids])/alpha
            return bool(np.all(raw>=lo) and np.all(raw<=hi) and np.all(abs(alpha*(raw-prev[ids]))<=step)),goal,raw
        edge=int(np.searchsorted(arc,s,side='right'))
        safe=s
        # Traverse every intervening corner. The first infeasible segment stops progress.
        while edge<len(arc):
            okay,goal,raw=valid(float(arc[edge]))
            if okay:
                safe=float(arc[edge]);edge+=1
                if edge==len(arc):break
            else:
                low,high=safe,float(arc[edge])
                for _ in range(45):
                    mid=(low+high)/2
                    if valid(mid)[0]:low=mid
                    else:high=mid
                safe=low;break
        assert safe>s+1e-12,(s,safe)
        okay,goal,raw=valid(safe);assert okay
        nxt=prev.copy();nxt[ids]=prev[ids]+np.clip(alpha*(np.clip(raw,lo,hi)-prev[ids]),-step,step)
        target=prev.copy();target[ids]=raw
        targets.append(target);applied.append(nxt);desired.append(goal);positions.append(safe);s=safe
    complete=bool(s==arc[-1]);applied=np.array(applied);desired=np.array(desired)
    t=np.arange(len(applied))*.02
    np.savez_compressed(O/'command_schedule.npz',t=t,source_arc=np.array(positions),applied=applied,
                        desired=desired,raw_target=np.array(targets),moving_ids=np.array(ids),lo=lo,hi=hi)
    path=read(D/'stance_com_path_v2_98/results.json')['dense']['path']
    counts={p:0 for p in ['shift','swing','approach','load']}
    for s in positions[1:]:counts[path[min(int(np.searchsorted(arc,s,side='left')),len(path)-1)]['phase']]+=1
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
             scope='One bounded CPUfloat32 command-only schedule along the SAME106 command polyline. Farthest feasible forward progress through each intervening segment at20ms, respecting inherited .03 desired clamp, alpha.1 and .003 increment cap. Up to1500command ticks; not physics ticks. No changed poses, forces, gains, filters or rates. Schedule is a constructed feasible arithmetic sequence, NOT a globally minimum duration or a realized physical trajectory. No new recorded motion.',
             completed=complete,command_ticks=len(targets),maximum_command_ticks=max_ticks,
             duration_s=float(t[-1]),allowance_s=13.5,time_gate=bool(complete and t[-1]<=13.5),
             phase_time_s={k:v*.02 for k,v in counts.items()},
             max_applied_vs_polyline_rad=float(max(abs(applied-desired).ravel())),
             final_applied_vs_endpoint_rad=float(max(abs(applied[-1]-Q[-1].astype(np.float32)))),
             physics_ticks=0,optimizer_updates=0,holdout_used=False,physical_admission=False,
             source_sha256={str(p.resolve()):sha(p) for p in [source,audit,Path(__file__)]})
    (O/'results.json').write_text(json.dumps(out,indent=2))
    (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
    print(json.dumps(out),flush=True)

if __name__=='__main__':main()
