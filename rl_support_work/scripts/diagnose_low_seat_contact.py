"""Recorded-data diagnosis only; simulator truth is never a runtime input."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

OLD=Path('/data/humanoid-sitting-perception/low_positive_transfer_01')
OUT=OLD.parent/'low_seat_contact_01/recorded_diagnosis_01.json'

def main():
    output=[]
    for bank in ['development_old','development_transfer','holdout']:
        folder=OLD/'ppo_01/seed_62111'/f'{bank}_parent'
        r=json.loads((folder/'results.json').read_text())
        geometry=json.loads((folder/'geometry.json').read_text())
        plans={a['case']:a for a in json.loads((folder/'rolling_plan_audit.json').read_text())}
        with np.load(folder/'trace.npz') as tf,np.load(folder/'every_tick.npz') as ef:
            t={k:tf[k] for k in tf.files};e={k:ef[k] for k in ef.files}
        names=r['body_names'];pelvis=names.index('pelvis');origins=np.asarray(r['origins'])
        for row in r['episodes']:
            i=row['case'];g=geometry[row['geometry_index']]
            if row['slip_ticks']==0 and not any(row['failures'].values()):continue
            loaded=np.linalg.norm(e['force'][:,i],axis=-1)>.01*e['weight'][:,i,None]
            slip=loaded & (e['slip'][:,i]>.02)
            ticks=np.flatnonzero(slip.any(-1));k=int(ticks[0]) if len(ticks) else None
            failures=np.flatnonzero(e['failure'][:,i]);event_time=float(e['t'][k]) if k is not None else float(e['t'][failures[0]])
            ti=int(np.argmin(abs(t['t']-event_time)))
            a=plans[i];u=np.clip((float(t['t'][ti])-10.)/5*24,0,24);lo=min(int(u),23);f=u-lo
            path=np.asarray(a['root_path']);target=(1-f)*path[lo,:3]+f*path[lo+1,:3]
            root=t['body'][ti,i,pelvis,:3]-origins[i]
            feet=[names.index(s+'_ankle_roll_link') for s in ['left','right']]
            foot_motion=float(np.linalg.norm(t['body'][:ti+1,i,feet,:3]-t['body'][0,i,feet,:3],axis=-1).max())
            output.append(dict(bank=bank,case=i,height=g['height'],slope=g['pitch_deg'],
                slip_ticks=row['slip_ticks'],failures=row['failures'],first_event_seconds=event_time,
                loaded_contact_slip_speeds=e['slip'][k,i].tolist() if k is not None else None,
                loaded_contacts=loaded[k].tolist() if k is not None else None,
                actual_minus_planned_root_m=(root-target).tolist(),
                predicted_endpoint_housing_clearance_m=a.get('model_clearance_path',[None])[-1],
                foot_displacement_before_event_m=foot_motion,
                fresh_geometry_at_event=bool(t['geometry'][ti,i,11]>0),
                actual_pelvis_pitch_rad=float(Rotation.from_quat(t['body'][ti,i,pelvis,3:]).as_euler('xyz')[1])))
    assert not OUT.exists()
    OUT.write_text(json.dumps(dict(scope='Offline recorded diagnosis, labels only; not a causal proof or physical clearance certificate',rows=output),indent=2))
    print('RECORDED_FAILURE_CASES',len(output))

if __name__=='__main__':main()
