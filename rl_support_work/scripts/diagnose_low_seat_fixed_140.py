"""Recorded-only failure timing and tracking diagnosis for rejected134."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'fixed_foot_probe_134/candidate';O=D/'fixed_diagnosis_140'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    O.mkdir(exist_ok=False);tr=np.load(P/'trace.npz');ticks=np.load(P/'every_tick.npz');rows=read(P/'results.json')['episodes'];records=read(P/'fixed_guard_records_132.json');plans=read(P/'fixed_plans_132.json');names=read(P/'static_inertial_model.json')['body_names'];pelvis=names.index('pelvis')
    geometry=read(P/'geometry.json');origins=np.array(read(P/'results.json')['origins']);out=[];series=[]
    for i,row in enumerate(rows):
        events={}
        for name,mask in [('slip',~ticks['no_slip'][:,i]),('forbidden_contact',ticks['wrong'][:,i]>10),('illegal_patch',~ticks['legal'][:,i]),('physical_failure',ticks['failure'][:,i])]:
            ids=np.flatnonzero(mask);events[name]=float(ticks['t'][ids[0]]) if len(ids) else None
        a=[r for r in records if r['case']==i];paused=[r for r in a if not r['progress_allowed']];p=next((r for r in plans if r['case']==i),None)
        item=dict(case=i,height_cm=geometry[row['geometry_index']]['height']*100,slope_deg=geometry[row['geometry_index']]['pitch_deg'],events=events,physical_flags=row['failures'],stable=row['success'],slip_ticks=row['slip_ticks'],illegal_ticks=row['illegal_ticks'],warning_ticks=row['traction_bad_ticks'],full_duration_s=row['duration_s'])
        if p:
            commands=np.array([e['load_checks'][0]['command'] for e in p['plan']['path']]);arc=np.r_[0,np.cumsum(np.max(abs(np.diff(commands,axis=0)),axis=1))];roots=np.array([e['root'] for e in p['plan']['path']]);pitch=Rotation.from_quat(roots[:,3:]).as_euler('xyz')[:,1]
            first=paused[0] if paused else None
            item.update(first_pause_time_s=first['tick']*.02 if first else None,first_pause=first,
                        progress_ticks=sum(r['progress_allowed'] for r in a),max_model_load_permission_bw=max(r['estimated_support']['max_normal_bw'] for r in a),
                        max_assumed_load_demand_bw=max(r['command_path'].get('demanded_normal_bw',0.) for r in a),
                        last_progress_fraction=a[-1]['command_path']['end']/arc[-1],parent_handoff_s=15.,
                        first_pause_precedes_first_physical_failure=bool(first and events['physical_failure'] is not None and first['tick']*.02<events['physical_failure']))
            for r in a:
                if r['tick']%5 or r['tick']>300:continue
                t=r['tick']*.02;at=int(np.flatnonzero(np.isclose(tr['t'],t,atol=1e-10))[0]);s=r['command_path']['start'];pr=np.array([np.interp(s,arc,roots[:,j]) for j in range(3)])
                actual=tr['body'][at,i,pelvis];R=Rotation.from_quat(actual[3:]).as_matrix();actual_pitch=Rotation.from_matrix(R).as_euler('xyz')[1]
                series.append(dict(case=i,t=t,planned_pitch_deg=float(np.rad2deg(np.interp(s,arc,pitch))),actual_pitch_deg=float(np.rad2deg(actual_pitch)),
                    planned_root=pr.tolist(),actual_root=(actual[:3]-origins[i]).tolist(),tracking_error_rad=r['tracking_error_rad'],geometry_available=r['measured_guard']['available'],progress_allowed=r['progress_allowed'],
                    foot_orientation_disagreement_deg=float(np.rad2deg(r['measured_guard']['foot_orientation_disagreement_rad'])) if r['measured_guard']['available'] else None))
        out.append(item)
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='All24 recorded episodes. First events use original20ms labels; body/IMU samples are0.1s and only bracket event geometry. Selected4 stopped on inconsistent foot orientation while still~260mm above the plane, before slips and physical failures. No modeled load was requested and frozenparent starts15s, after these failures. This locates an early support/tracking failure; it is not single-cause proof. A paused static-reference command did not stabilize the actual motion. No threshold relaxation, rerun or new policy.',
                rows=out,tracking_series=series,physics_ticks=0,optimizer_updates=0,physical_repair=False,
                source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),P/'trace.npz',P/'every_tick.npz',P/'fixed_guard_records_132.json',P/'fixed_plans_132.json',P/'results.json']})
    (O/'recorded_diagnosis.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
    print(json.dumps([dict(case=r['case'],first_pause=r.get('first_pause_time_s'),events=r['events'],load_demand=r.get('max_assumed_load_demand_bw')) for r in out if 'first_pause_time_s' in r]),flush=True)
if __name__=='__main__':main()
