"""Recorded-only force-allocation and early balance diagnosis; no simulation."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
from scipy.spatial.transform import Rotation
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'fixed_foot_probe_134/candidate';B=D/'normal_landing3s_06/candidate';O=D/'force_jump_diagnosis_143'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def foot_stats(e):
    l=e['load_checks'][0];w=np.array(l['corner_fractions']);x=np.array(l['corner_world_points']);side=np.array(l['corner_sides']);out=[]
    for s in ['left','right']:
        use=side==s;f=w[use].sum();cop=(w[use,None]*x[use]).sum(0)/f
        anchor=np.array(e['foot_targets'][s]['position']);moment=np.sum(np.cross(x[use]-anchor,np.c_[np.zeros((sum(use),2)),w[use]]),axis=0)
        out.append(dict(vertical_bw=float(f),cop_world_m=cop.tolist(),moment_about_ankle_per_BW_m=moment.tolist()))
    return out
def main():
    O.mkdir(exist_ok=False);plans=read(P/'fixed_plans_132.json');records=read(P/'fixed_guard_records_132.json');paths=[];series=[];all_rows=[];sources=[Path(__file__).resolve(),P/'fixed_plans_132.json',P/'fixed_guard_records_132.json']
    planmap={p['case']:p['plan'] for p in plans};recordmap={(r['case'],r['tick']):r for r in records}
    for case,p in planmap.items():
        ids=p['moving_ids'];Q=np.array([e['q'] for e in p['path']]);C=np.array([e['load_checks'][0]['command'] for e in p['path']]);T=np.array([e['load_checks'][0]['static_torque_Nm'] for e in p['path']]);F=[foot_stats(e) for e in p['path']]
        paths.append(dict(case=case,first_pose_change_max_rad=float(abs(Q[1]-Q[0]).max()),first_command_change_max_rad=float(abs(C[1]-C[0]).max()),first_static_torque_change_Nm=(T[1]-T[0]).tolist(),
            first_ankle_pitch_torque_Nm=T[:2,[4,10]].tolist(),first_foot_force=F[:2],max_adjacent_command_change_rad=float(abs(np.diff(C,axis=0)).max()),
            max_adjacent_pose_change_rad=float(abs(np.diff(Q,axis=0)).max()),max_adjacent_ankle_torque_change_Nm=float(abs(np.diff(T[:,[4,10]],axis=0)).max()),
            full_model_foot_stats=F,static_torque_Nm=T.tolist(),moving_ids=ids))
    for label,folder in [('rejected134',P),('best_normal3s',B)]:
        tr=np.load(folder/'trace.npz');ev=np.load(folder/'every_tick.npz');ob=np.load(folder/'shear_observer.npz');rows=read(folder/'results.json')['episodes'];names=read(folder/'static_inertial_model.json')['body_names'];namesq=read(folder/'results.json')['joint_names']
        sources += [folder/f for f in ['trace.npz','every_tick.npz','shear_observer.npz','results.json','static_inertial_model.json']]
        footids=[names.index(s+'_ankle_roll_link') for s in ['left','right']];ankles=[namesq.index(s+'_ankle_pitch_joint') for s in ['left','right']];pelvis=names.index('pelvis')
        for i,row in enumerate(rows):
            indexes=np.flatnonzero((tr['t']>=1)&(tr['t']<=5));all_rows.append(dict(source=label,case=i,selected_in134=i in planmap,stable=row['success'],slip_ticks=row['slip_ticks'],failures=row['failures']))
            for k in indexes:
                t=float(tr['t'][k]);et=int(np.flatnonzero(np.isclose(ev['t'],t,atol=1e-10))[0]);ot=int(np.flatnonzero(np.isclose(ob['t'],t,atol=1e-10))[0]);weight=float(ev['weight'][et,i]);r=dict(source=label,case=i,t=t,
                    actual_foot_forces_bw=(ev['force'][et,i,5:7]/weight).tolist(),actual_ankle_pitch_torque_Nm=tr['torque'][k,i,ankles].tolist(),
                    estimated_foot_wrench_per_BW=(ob['filtered'][ot,i]/weight).tolist(),observer_condition=ob['condition'][ot,i].tolist(),
                    actual_foot_pitch_deg=Rotation.from_quat(tr['body'][k,i,footids,3:]).as_euler('xyz',degrees=True)[:,1].tolist(),
                    actual_pelvis_pitch_deg=float(Rotation.from_quat(tr['body'][k,i,pelvis,3:]).as_euler('xyz',degrees=True)[1]),
                    loaded_slip=not bool(ev['no_slip'][et,i]),physical_failure=bool(ev['failure'][et,i]),geometry_valid=bool(tr['geometry'][k,i,11]),geometry_age_s=float(tr['geometry'][k,i,12]))
                if label=='rejected134' and i in planmap and (i,round(t/.02)) in recordmap:
                    p=planmap[i];rr=recordmap[i,round(t/.02)];C=np.array([e['load_checks'][0]['command'] for e in p['path']]);arc=np.r_[0,np.cumsum(np.max(abs(np.diff(C,axis=0)),axis=1))];s=rr['command_path']['start'];tau=np.array([e['load_checks'][0]['static_torque_Nm'] for e in p['path']]);F=next(x for x in paths if x['case']==i)['full_model_foot_stats']
                    r.update(progress=rr['progress_allowed'],planned_ankle_pitch_torque_Nm=[float(np.interp(s,arc,tau[:,j])) for j in [4,10]],planned_foot_vertical_bw=[float(np.interp(s,arc,[f[j]['vertical_bw'] for f in F])) for j in [0,1]],
                        planned_foot_moment_y_per_BW_m=[float(np.interp(s,arc,[f[j]['moment_about_ankle_per_BW_m'][1] for f in F])) for j in [0,1]],tracking_error_rad=rr['tracking_error_rad'])
                series.append(r)
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),physics_ticks=0,optimizer_updates=0,physical_repair=False,
        scope='All48 saved full-trial outcomes; recorded1–5s samples compare rejected134 and bestnormal3s. Model allocation abruptly imposes opposing foot moments despite tiny first pose change. Actual contact forces and torques are evaluation diagnostics only. Saved contact_position is not treated as an aggregate center-of-pressure measurement. Static wrench estimates can be biased by dynamics. This is not sole-cause proof or repair admission.',
        paths=paths,rows=all_rows,series=series,source_sha256={str(p):sha(p) for p in sources})
    (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
    print(json.dumps([{k:r[k] for k in ['case','first_pose_change_max_rad','first_command_change_max_rad','first_ankle_pitch_torque_Nm','max_adjacent_ankle_torque_change_Nm']} for r in paths]),flush=True)
if __name__=='__main__':main()
