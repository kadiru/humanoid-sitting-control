"""Recorded-label diagnosis of landing timing; no runtime truth inputs."""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def read(p):return json.loads(p.read_text())
def main():
    output={}
    for variant in ['normal_landing3s_06','normal_counterbalance_13']:
        p=D/variant/'candidate';r=read(p/'results.json');geometry=read(p/'geometry.json')
        plans={a['case']:a for a in read(p/'rolling_plan_audit.json')}
        t=np.load(p/'trace.npz');e=np.load(p/'every_tick.npz');origins=np.asarray(r['origins']);names=r['body_names']
        pelvis=names.index('pelvis');feet=[names.index(s+'_ankle_roll_link') for s in ['left','right']];rows=[]
        for row in r['episodes']:
            i=row['case'];g=geometry[row['geometry_index']];a=plans[i]
            loaded=np.linalg.norm(e['force'][:,i],axis=-1)>.01*e['weight'][:,i,None]
            slips=loaded & (e['slip'][:,i]>.02)
            first=lambda mask:float(e['t'][np.flatnonzero(mask)[0]]) if np.any(mask) else None
            contact=first(loaded[:,:5].any(-1));slip=first(slips.any(-1));forbidden=first(e['wrong'][:,i]>10.)
            event=min([v for v in [contact,slip,forbidden] if v is not None],default=20.)
            idx=t['t']<=event;root=t['body'][:,i,pelvis,:3]-origins[i]
            u=np.clip((t['t']-10)/5*24,0,24);lo=np.minimum(u.astype(int),23);f=u-lo
            path=np.asarray(a['root_path']);target=(1-f[:,None])*path[lo,:3]+f[:,None]*path[lo+1,:3]
            samples=[]
            for s in sorted(set([10.,11.,12.,13.,15.,20.,round(event,1)])):
                ti=int(np.argmin(abs(t['t']-s)))
                samples.append(dict(t=float(t['t'][ti]),root=root[ti].tolist(),error=(root[ti]-target[ti]).tolist(),
                    pitch=float(Rotation.from_quat(t['body'][ti,i,pelvis,3:]).as_euler('xyz')[1]),
                    foot_pitch=Rotation.from_quat(t['body'][ti,i,feet,3:]).as_euler('xyz')[:,1].tolist()))
            rows.append(dict(case=i,height=g['height'],slope=g['pitch_deg'],stable=row['success'],
                slip_ticks=row['slip_ticks'],illegal_ticks=row['illegal_ticks'],failures=row['failures'],
                first_seat_load=contact,first_slip=slip,first_forbidden=forbidden,
                slip_by_contact=slips.sum(0).tolist(),peak_before_event_foot_displacement_m=float(np.linalg.norm(t['body'][idx,i][:,feet,:3]-t['body'][0,i,feet,:3],axis=-1).max()),
                fresh_geometry_all_before_event=bool((t['geometry'][idx,i,11]>0).all()),
                measured_normal_error_deg=float(np.rad2deg(np.arccos(np.clip(np.asarray(a['normal'])@np.asarray(g['normal']),-1,1)))),
                measured_offset_error_m=float(a['offset']+np.asarray(g['normal'])@np.asarray(g['top_center'])),
                planned_initial_pitch=a['pitch'],planned_final_housing_clearance=a.get('model_clearance_path',[None])[-1],samples=samples))
        output[variant]=rows
    dest=D/'counterbalance_diagnosis_13.json';assert not dest.exists()
    dest.write_text(json.dumps(dict(scope='Recorded-label diagnosis only; no single-factor causal proof or clearance certificate',variants=output),indent=2))
    for variant,rows in output.items():
        print(variant)
        for r in rows[::2]:print(json.dumps({k:v for k,v in r.items() if k!='samples'}))
if __name__=='__main__':main()
