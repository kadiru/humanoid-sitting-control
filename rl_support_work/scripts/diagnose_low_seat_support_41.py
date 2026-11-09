"""Read-only recorded support/clearance diagnosis; never changes physical gates."""
from pathlib import Path
import json, hashlib, math
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from humanoid_sitting.shared_frame import BodyModel, apply, transform
from humanoid_sitting.resting_objective import RestingObjectiveCfg
from plan_seat_load_targets import clipped_vertices
from audit_low_seat_contact_probe_v2 import clean

D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
OUT=D/'support_diagnosis_41'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(exist_ok=False)
    cfg=RestingObjectiveCfg(); assert cfg.maximum_trunk_tilt_deg==40. and cfg.minimum_rest_normal_load_bw==.05
    model=BodyModel('runs/g1_kinematics_geometry_04/model.json')
    sources=[Path('sitting_task/src/humanoid_sitting/resting_objective.py')]+[Path('scripts')/n for n in ['run_supported_postures.py','run_supported_transition.py','run_contact_effort_repair.py','run_progressive_seat_ppo.py','run_crouch_rest_safe_ppo.py','run_crouch_rest_audited_ppo.py','run_traction_transfer_ppo.py','run_posture_episode_ppo.py','run_positive_slope_ppo_v4.py','run_low_positive_feedback_v2.py','low_seat_staged_planner_36.py']]
    selected=[0,1,2,3,6,7,10,11]; variants=['normal_landing3s_06','joint_path_17','tracking_headroom_31','staged_approach_36','dual_view_approach_40']
    rows=[]; series={}
    for variant in variants:
        p=D/variant/'candidate'; result=read(p/'results.json'); geoms=read(p/'geometry.json'); tr=np.load(p/'trace.npz'); ev=np.load(p/'every_tick.npz')
        plans={r['case']:r for r in read(p/'rolling_plan_audit.json')}; names=result['body_names']; origins=np.asarray(result['origins']); series[variant]={}
        for r in result['episodes']:
            i=r['case']; geo=geoms[r['geometry_index']]; late=tr['t']>15.; hits=np.flatnonzero(ev['wrong'][:,i]>10); event=float(ev['t'][hits[0]]) if len(hits) else None
            tilt=np.rad2deg(tr['tilt'][:,i]); support=tr['support'][:,i]
            row=dict(variant=variant,case=i,height_cm=geo['height']*100,slope_deg=geo['pitch_deg'],stable=r['success'],clean=clean(r),final_hold_s=r['final_hold_s'],first_housing_s=event,
                final5_sample_count=int(late.sum()),final5_tilt_deg_min=float(tilt[late].min()),final5_tilt_deg_max=float(tilt[late].max()),final5_samples_above_existing40deg=int((tilt[late]>40).sum()),
                final5_samples_below_existing_support005bw=int((support[late]<.05).sum()),final_support_bw=float(support[-1]),maximum_support_bw=float(support.max()),sampled_good_ticks=int(tr['good'][:,i].sum()),samples=[])
            plan=plans.get(i,{}); roots=np.asarray(plan.get('root_path',[])); staged=plan.get('staged_36')
            if len(roots):
                target=roots[-1]; planned_tilt=math.degrees(math.acos(np.clip(Rotation.from_quat(target[3:]).as_matrix()[2,2],-1,1)))
                row.update(planned_final_tilt_deg=planned_tilt,planned_final_tilt_meets_existing_support_criterion=planned_tilt<=40.)
            if i in selected and variant in ['staged_approach_36','dual_view_approach_40']:
                phase=np.load(p/'staged_ticks_36.npz'); phase_t=phase['t'][:,i]; elapsed=phase['elapsed_before'][:,i]
                seatR=Rotation.from_quat(geo['rotation_xyzw']).as_matrix(); center=np.asarray(geo['top_center'])
                times=[1.,8.,12.,13.,14.,15.,16.,20.]
                if event is not None: times.extend([np.floor(event*10)/10, np.ceil(event*10)/10])
                for time in sorted(set(times)):
                    k=int(np.argmin(abs(tr['t']-time))); body=tr['body'][k,i]; root=body[names.index('pelvis')].copy(); root[:3]-=origins[i]
                    u=np.clip(np.interp(time,phase_t,elapsed)/14*24,0,24); lo=min(int(u),23); f=u-lo; planned_xyz=roots[lo,:3]*(1-f)+roots[lo+1,:3]*f
                    mesh=[]
                    for name in ['pelvis','left_hip_pitch_link','right_hip_pitch_link']:
                        pose=body[names.index(name)]; T=transform(Rotation.from_quat(pose[3:]).as_matrix(),pose[:3]-origins[i]); vertices=(apply(T,model.vertices[name])-center)@seatR; vertices[:,0]-=.4; clipped=clipped_vertices(vertices)
                        mesh.append(dict(body=name,min_finite_seat_clearance_mm=float(clipped[:,2].min()*1000) if len(clipped) else None))
                    feet=[]
                    for side in ['left','right']:
                        pose=body[names.index(side+'_ankle_roll_link')]; feet.append(dict(side=side,pitch_deg=float(Rotation.from_quat(pose[3:]).as_euler('xyz',degrees=True)[1])))
                    geometry=tr['geometry'][k,i]
                    row['samples'].append(dict(t=float(tr['t'][k]),path_elapsed_s=float(np.interp(time,phase_t,elapsed)),root_xyz_m=root[:3].tolist(),planned_xyz_m=planned_xyz.tolist(),root_error_mm=((root[:3]-planned_xyz)*1000).tolist(),tilt_deg=float(tilt[k]),seat_support_bw=float(support[k]),geometry_valid=bool(geometry[11]),geometry_age_s=float(geometry[12]),mesh=mesh,feet=feet))
                series[variant][i]=dict(t=tr['t'].tolist(),tilt_deg=tilt.tolist(),support_bw=support.tolist(),elapsed=np.interp(tr['t'],phase_t,elapsed).tolist(),first_housing_s=event)
            rows.append(row)
    out=dict(scope='Read-only saved full20s evidence. Pose samples0.1s bracket0.02s contacts; no exact-event geometry or sole-cause claim.',criteria_unchanged=True,criterion_source_sha256={str(p.resolve()):sha(p) for p in sources},
        stable_support=dict(max_tilt_deg=40.,min_rest_normal_load_bw=.05,max_speed_m_s=cfg.maximum_speed_m_s,max_angular_speed_rad_s=cfg.maximum_angular_speed_rad_s),
        planner_omission='The staged36/40 .9rad endpoint is51.566deg: exceeds unchanged40deg qualifying-support tilt. Its model checks included clearance/rate/foot support but omitted this final-posture requirement. The55deg fall limit is a different criterion. Original physical failures and audits remain unchanged.',
        implication='Even perfect .9rad tracking cannot qualify for the existing final5s support gate. This is a planning defect, not evidence that changing the physical criterion is warranted. Actual housing contact and inadequate seat loading are additional independent failures.',
        rows=rows)
    (OUT/'recorded_support.json').write_text(json.dumps(out,indent=2));(OUT/'series.json').write_text(json.dumps(series))
    fig,axes=plt.subplots(2,4,figsize=(15,6),sharex=True)
    for col,i in enumerate([0,2,6,10]):
        for variant,color,label in [('staged_approach_36','#999999','Rear only36'),('dual_view_approach_40','#286a93','Current fallback40')]:
            s=series[variant][i];axes[0,col].plot(s['t'],s['tilt_deg'],color=color,label=label);axes[1,col].plot(s['t'],s['support_bw'],color=color,label=label)
            if s['first_housing_s'] is not None:
                for ax in axes[:,col]:ax.axvline(s['first_housing_s'],color=color,ls=':',alpha=.8)
        row=next(r for r in rows if r['variant']=='dual_view_approach_40' and r['case']==i)
        axes[0,col].set_title(f"{row['height_cm']:g}cm / +{row['slope_deg']:g}°")
        axes[0,col].axhline(40,color='#b33c39',ls='--',label='Existing support tilt limit')
        axes[0,col].axhline(np.rad2deg(.9),color='#b5832a',ls='-.',label='Planned final tilt')
        axes[1,col].axhline(.05,color='#b33c39',ls='--',label='Existing support minimum')
        for ax in axes[:,col]:ax.grid(alpha=.2);ax.set_xlim(0,20);ax.axvspan(15,20,color='#ddd',alpha=.15)
        axes[1,col].set_xlabel('Time(s)')
    axes[0,0].set_ylabel('Recorded pelvis tilt (degrees)');axes[1,0].set_ylabel('Legal seat normal load / BW')
    axes[0,0].legend(fontsize=7);axes[1,0].legend(fontsize=7)
    fig.suptitle('The current view keeps motion moving; the planned resting posture is inadmissible\nExisting40° support criterion and legal loading are unchanged; dotted lines mark housing contacts')
    fig.tight_layout();fig.savefig(OUT/'support_and_tilt.png',dpi=160);plt.close(fig)
    (OUT/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in OUT.iterdir() if p.is_file()},indent=2))
    print(json.dumps([r for r in rows if r['variant']=='dual_view_approach_40' and r['case'] in selected],indent=2))
if __name__=='__main__':main()
