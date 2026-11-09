"""Recorded-only COM, foot support and force-realization diagnosis.

Truth-derived COM is evaluation only. The separately reconstructed estimator
uses saved encoders/IMU and initial kinematic foot anchors. No new physics.
"""
from pathlib import Path
import datetime, hashlib, json
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel, apply, level_rotation, transform

D = Path('/data/humanoid-sitting-perception/low_seat_contact_01')
O = D/'balance_com_diagnosis_170'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def first(t, mask):
    ids = np.flatnonzero(mask)
    return float(t[ids[0]]) if len(ids) else None

def main():
    O.mkdir(exist_ok=False)
    mp=D/'cooked_model_50/model.json'
    ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json'
    model=BodyModel(mp); inertia=read(ip); masses=np.array(inertia['mass']); comlocal=np.array(inertia['local_com'])
    all_rows=[]; sources={str(Path(__file__).resolve()):sha(__file__), str(mp):sha(mp), str(ip):sha(ip)}
    for variant in ['balance_feedback_probe_164']:
        p=D/variant/'candidate'; tr=dict(np.load(p/'trace.npz')); e=dict(np.load(p/'every_tick.npz'))
        sen=np.load(p/'sensor_inputs.npz')['sensors'][0]; obs=dict(np.load(p/'shear_observer.npz'))
        result=read(p/'results.json'); names=result['body_names']; origins=np.array(result['origins'])
        assert names==inertia['body_names']
        plans=read(p/'fixed_plans_132.json') if (p/'fixed_plans_132.json').exists() else []
        records=read(p/'fixed_guard_records_132.json') if plans else []
        lookup={(r['case'],r['tick']):r for r in records}
        for n in ['trace.npz','every_tick.npz','sensor_inputs.npz','shear_observer.npz','results.json']:
            sources[str(p/n)]=sha(p/n)
        for row in result['episodes']:
            case=row['case']; tt=tr['t']; plan=next((r['plan'] for r in plans if r['case']==case),None)
            anchor=None; series=[]
            if plan:
                anchor=plan['path'][0]['foot_targets']; roots=np.array([r['root'] for r in plan['path']])
                Q=np.array([r['load_checks'][0]['command'] for r in plan['path']]); arc=np.r_[0,np.cumsum(abs(np.diff(Q,axis=0)).max(1))]
                model_com=np.array([r['com'] for r in plan['path']]); model_tau=np.array([r['load_checks'][0]['static_torque_Nm'] for r in plan['path']])
            # Evaluate every saved sample, including failures. No fragment effort claim.
            for k,t in enumerate(tt):
                tick=round(t/.02); si=round(t/.1); ei=tick-1; q=tr['q'][k,case]; gravity=sen[si,case,339:342] if si<len(sen) else None
                if gravity is not None: assert np.array_equal(sen[si,case,270:293], q[:23]), 'Actual414 encoder layout changed'
                poses=tr['body'][k,case]; Rbody=Rotation.from_quat(poses[:,3:]).as_matrix()
                actual=(poses[:,:3]-origins[case]+np.einsum('bij,bj->bi',Rbody,comlocal))
                C=np.average(actual,axis=0,weights=masses)
                support=[]; feet=[]
                for side in ['left','right']:
                    name=side+'_ankle_roll_link'; bi=names.index(name); v=model.vertices[name]
                    sole=v[v[:,2]<v[:,2].min()+.001]
                    points=sole@Rbody[bi].T+poses[bi,:3]-origins[case]
                    support.extend(points[:,:2]); feet.append(points)
                hull=ConvexHull(support).equations; margin=float(np.min(-(hull[:,:2]@C[:2]+hull[:,2])))
                R=level_rotation(gravity) if gravity is not None else None; fk=model.fk(q)
                relcom=np.average([apply(fk[b],cc) for b,cc in zip(names,comlocal)],axis=0,weights=masses)
                estimated=None; estimate_error=None; foot_imu_pitch=[]
                for side in (['left','right'] if R is not None else []):
                    foot_imu_pitch.append(float(Rotation.from_matrix(R@fk[side+'_ankle_roll_link'][:3,:3]).as_euler('xyz',degrees=True)[1]))
                if anchor and R is not None:
                    root=np.mean([np.array(anchor[s]['position'])-R@fk[s+'_ankle_roll_link'][:3,3] for s in ['left','right']],axis=0)
                    estimated=root+R@relcom; estimate_error=float(np.linalg.norm(estimated-C))
                info=dict(t=float(t),actual_com_m=C.tolist(),actual_support_margin_m=margin,
                    actual_foot_vertical_bw=(e['force'][ei,case,5:7,2]/e['weight'][ei,case]).tolist(),
                    actual_foot_horizontal_bw=(e['force'][ei,case,5:7,0]/e['weight'][ei,case]).tolist(),
                    estimated_foot_wrench=obs['filtered'][si,case].tolist(),observer_condition=obs['condition'][si,case].tolist(),
                    encoder_imu_foot_pitch_deg=foot_imu_pitch,encoder_imu_com_m=None if estimated is None else estimated.tolist(),
                    encoder_imu_com_error_m=estimate_error,actual_pelvis_pitch_deg=float(Rotation.from_matrix(Rbody[names.index('pelvis')]).as_euler('xyz',degrees=True)[1]),
                    geometry_valid=bool(tr['geometry'][k,case,11]),geometry_age_s=float(tr['geometry'][k,case,12]),
                    no_slip=bool(e['no_slip'][ei,case]),physical_failure=bool(e['failure'][ei,case]))
                rec=lookup.get((case,tick))
                if rec:
                    pos=rec['command_path']['start']; desired=np.array([np.interp(pos,arc,model_com[:,j]) for j in range(3)])
                    info.update(planned_com_m=desired.tolist(),planned_com_error_m=(desired-C).tolist(),progress_allowed=rec['progress_allowed'],
                        model_ankle_torque_Nm=[float(np.interp(pos,arc,model_tau[:,j])) for j in [4,10]],
                        actual_ankle_torque_Nm=tr['torque'][k,case,[model.names.index(s+'_ankle_pitch_joint') for s in ['left','right']]].tolist())
                series.append(info)
            com=np.array([r['actual_com_m'] for r in series]); velocity=np.gradient(com,tt,axis=0); height=com[:,2]
            # LIP capture point is a diagnostic approximation, not a physical criterion.
            for k,r in enumerate(series):
                r['recorded_com_velocity_m_s']=velocity[k].tolist()
                r['approx_capture_point_x_m']=float(com[k,0]+velocity[k,0]/np.sqrt(9.81/max(height[k],.1)))
            item=dict(variant=variant,case=case,stable=row['success'],failures=row['failures'],slip_ticks=row['slip_ticks'],
                first_com_outside_actual_support_sample_s=first(tt,np.array([r['actual_support_margin_m'] for r in series])<0),
                first_slip_s=first(e['t'],~e['no_slip'][:,case]),first_physical_s=first(e['t'],e['failure'][:,case]),series=series)
            if plan:
                valid=[r for r in series if 1<=r['t']<=6.5 and r['encoder_imu_com_error_m'] is not None]
                item['first_1_to_6p5s_max_encoder_imu_com_error_m']=max(r['encoder_imu_com_error_m'] for r in valid)
            all_rows.append(item)
        print(variant,'recorded episodes',len(result['episodes']),flush=True)
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,rows=all_rows,source_sha256=sources,
        physics_ticks=0,optimizer_updates=0,physical_admission=False,
        limitations='COM velocity uses0.1s finite differences; LIP capture point is approximate. Foot wrench is quasi-static. Support hull is geometric, not friction/dynamic certification. Encoder/IMU COM assumes initial feet stationary and yaw unobserved; validate against truth for diagnosis only. Body COM uses stored body-frame transforms plus frozen localCOM. No aggregate actual CoP claimed.')
    (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
    print(json.dumps([{k:r.get(k) for k in ['case','first_com_outside_actual_support_sample_s','first_slip_s','first_physical_s','first_1_to_6p5s_max_encoder_imu_com_error_m']} for r in all_rows if r['variant']=='balance_feedback_probe_164' and r['case'] in [0,1,6,7]]),flush=True)
if __name__=='__main__':main()
