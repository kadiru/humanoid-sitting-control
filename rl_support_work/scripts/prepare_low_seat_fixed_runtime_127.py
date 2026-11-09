"""No-physics preparation of a current-input planner and support observer.

Saved measurements are fixtures ONLY in this preparation script. The planner
accepts arrays and static models, and cannot read saved trials or case labels.
"""
from pathlib import Path
import datetime, hashlib, json
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel
from low_seat_fixed_foot_plan_126 import build_plan, measured, route
from foot_wrench_observer import FootWrenchObserver

D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
O=D/'fixed_runtime_preparation_127'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    O.mkdir(exist_ok=False)
    model_path=D/'cooked_model_50/model.json'
    inertia_path=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json'
    cal_path=D.parent/'contact_effort_repair_01/plane_calibration.json'
    model=BodyModel(model_path); meta=read(model_path); inertia=read(inertia_path);cal=read(cal_path)
    config=dict(routes=[])
    for c,height,slope in [(0,[.357,.363],[14.5,15.5]),(6,[.367,.373],[15.75,16.75])]:
        p=read(D/'spatial_contact_screen_v2_58'/f'case_{c}.json')['best']['parameters'][:5]
        p[0]-=.001
        config['routes'].append(dict(name=f'measured_route_{c}',height_m=height,slope_deg=slope,endpoint=p))
    config.update(scope='Two development-only measured geometry routes; no truth case index in planner. No broad generalization claim.',
                  physics_admitted=False,normal_load_max_bw=.1,desired_margin_rad=.03,alpha=.1,command_rate_rad_s=.15)
    (O/'candidate_config.json').write_text(json.dumps(config,indent=2))
    trace_path=D/'normal_landing3s_06/candidate/trace.npz'; sensor_path=D/'normal_landing3s_06/candidate/sensor_inputs.npz'
    tr=np.load(trace_path); ss=np.load(sensor_path)['sensors']; at=int(np.argmin(abs(tr['t']-1)))
    observer=FootWrenchObserver(model,inertia)
    selected=[]; routes=[]; rows=[]; static_observer=[]
    for c in range(24):
        q=tr['q'][at,c].astype(float); g=ss[0,10,c,339:342]; geometry=tr['geometry'][at,c]
        assert np.array_equal(q[:23].astype(np.float32),ss[0,10,c,270:293])
        reference=[tr['estimated_root_x'][at,c],0,0]
        plane=measured(model,q,g,geometry,reference,cal)
        chosen=route(plane[3],plane[4],config)
        routes.append(dict(case=c,measured_height_m=plane[3],measured_slope_deg=plane[4],route=chosen['name'] if chosen else None))
        if chosen is None:continue
        selected.append(c)
        result=build_plan(model,meta,inertia,cal,q,g,geometry,reference,tr['command'][at,c],chosen['endpoint'])
        (O/f'plan_{c}.json').write_text(json.dumps(result,indent=2))
        parity=None
        if c in [0,6]:
            old=read(D/'staged_fixed_foot_contact_120'/f'case_{c}/results.json')
            parity={key:float(np.max(abs(np.array([p[key] for p in result['path']])-np.array([p[key] for p in old['path']]))))
                    for key in ['q','root','clearance_m','foot_position_error_m','foot_orientation_error_rad','min_command_margin_rad']}
            parity['commands']=float(np.max(abs(np.array([p['load_checks'][0]['command'] for p in result['path']])-np.array([p['load_checks'][0]['command'] for p in old['path']]))))
            assert max(parity.values())<1e-9,parity
        for k,e in enumerate(result['path']):
            load=e['load_checks'][0]; torque=np.zeros(37);torque[result['moving_ids']]=load['static_torque_Nm']
            R=Rotation.from_quat(e['root'][3:]).as_matrix();gravity=R.T@np.array([0.,0.,-1.])
            wrench,cond=observer.estimate(np.array(e['q']),torque,gravity)
            estimated=1-wrench[:,1].sum()/observer.weight
            expected=load['vertical_seat_fraction']
            static_observer.append(dict(case=c,knot=k,phase=e['phase'],estimated_vertical_seat_bw=float(estimated),
                                        assumed_vertical_seat_bw=expected,error_bw=float(estimated-expected),max_condition=float(max(cond))))
        rows.append(dict(case=c,summary=result['summary'],model_passed=result['model_passed'],old120_parity=parity))
        print(json.dumps(rows[-1]),flush=True)
    assert selected==[0,1,6,7],selected
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Planner extraction/current-input preparation only. No runtime installed; no physical admission. Sagittal observer checked against assumed full spatial model forces, not contact truth.',
                selected_cases=selected,routes=routes,plans=rows,planner_replay_passed=True,
                all_selected_paths_passed=all(r['model_passed'] for r in rows),
                maximum_sagittal_observer_static_error_bw=max(abs(r['error_bw']) for r in static_observer),
                maximum_sagittal_observer_condition=max(r['max_condition'] for r in static_observer),
                observer_rows=static_observer,physics_ticks=0,optimizer_updates=0,holdout_used=False,physical_admission=False,
                source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),Path(__file__).with_name('low_seat_fixed_foot_plan_126.py').resolve(),model_path,inertia_path,cal_path,trace_path,sensor_path]})
    (O/'results.json').write_text(json.dumps(result,indent=2))
    (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['routes','plans','observer_rows','source_sha256']}),flush=True)
if __name__=='__main__':main()
