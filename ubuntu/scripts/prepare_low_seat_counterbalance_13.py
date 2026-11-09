"""Recorded development-state model checks for one bounded counterbalance probe."""
import json,copy,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from plan_seat_load_targets import clipped_vertices
import low_seat_normal_timing_controller as normal
import low_seat_counterbalance_controller as changed
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def main():
    p=D/'normal_landing3s_06/candidate';read=lambda n:json.loads((p/n).read_text())
    m=BodyModel('runs/g1_kinematics_geometry_04/model.json');inertial=read('static_inertial_model.json')
    paths=np.load(p/'last_rolling_paths.npz');t=np.load(p/'trace.npz');plans=read('rolling_plan_audit.json')
    cfg=json.loads((D/'normal_landing3s_guarded_09.json').read_text());idx=int(np.argmin(abs(t['t']-10.)))
    rows=[]
    for a in plans:
        i=a['case'];q=t['q'][idx,i];command=t['command'][idx,i]
        R=Rotation.from_quat(t['body'][idx,i,0,3:]).as_matrix();gravity=R.T@np.array([0.,0.,-1.])
        normal.previous=lambda *args,**kw:(paths['command'][i].copy(),paths['q'][i].copy(),copy.deepcopy(a))
        args=(m,q,gravity,None,None,command,None)
        before,bq,ba=normal.plan(*args,inertial=inertial,config=cfg)
        cs,qs,b=changed.plan(*args,inertial=inertial,config=cfg)
        assert np.isfinite(cs).all() and np.isfinite(qs).all() and b['ik_error']<.003
        assert np.array_equal(cs[0],command)
        active=b['counterbalance_probe']['applied'];arm=b['counterbalance_probe'].get('arm_joint_ids',[])
        assert np.array_equal(np.asarray(b['root_path']),np.asarray(ba['root_path']))
        if active:
            other=[j for j in range(37) if j not in arm]
            assert np.array_equal(qs[:,other],bq[:,other])
            delta=cs[:,arm]-before[:,arm]
            assert abs(delta).max()<=.3500001
            rate=float(abs(np.diff(cs[:,arm],axis=0)).max()*24/5)
            assert rate<=.150001,rate
            assert b['counterbalance_probe']['model_path'][-1]['after_com_x']>b['counterbalance_probe']['model_path'][-1]['before_com_x']
            n=np.array(b['normal']);SR=Rotation.from_euler('y',np.arctan2(n[0],n[2])).as_matrix();center=np.array([-.4,0.,(n[0]*.4-b['offset'])/n[2]])
            clear=[]
            for qi,root in zip(qs,b['root_path']):
                root=np.asarray(root);T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);f=m.fk(qi)
                for name in m.vertices:
                    if not any(s in name for s in ['shoulder','elbow','hand','zero','one','two','three','four','five','six']):continue
                    v=(apply(T@f[name],m.vertices[name])-center)@SR;v[:,0]-=.4;v=clipped_vertices(v)
                    if len(v):clear.append(float(v[:,2].min()))
            assert not clear or min(clear)>.02
        else:
            assert np.array_equal(cs,before) and np.array_equal(qs,bq)
            rate=0.;clear=[]
        rows.append(dict(case=i,active=active,first_command_exact=True,unselected_exact=not active,
            root_and_leg_pose_path_unchanged=True,arm_peak_target_rate_rad_s=rate,
            min_model_arm_seat_clearance_m=min(clear) if clear else None,
            model=b['counterbalance_probe'],ik_error=b['ik_error']))
    out=D/'counterbalance_screen_13.json';assert not out.exists()
    out.write_text(json.dumps(dict(passed=True,scope='Offline development input/model check only, not physical admission. A reference-arm change tests forward COM shift; housing margin remains narrow. Controls use measured sensors/model only.',
        original_reliable_approach_seconds=10,physical_trials=0,optimizer_updates=0,fresh_holdout=False,
        sources={str(Path(__file__)):hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},rows=rows),indent=2))
    print(json.dumps(dict(passed=True,active=sum(r['active'] for r in rows),
        max_arm_rate=max(r['arm_peak_target_rate_rad_s'] for r in rows),
        com_shift=[r['model']['model_path'][-1]['after_com_x']-r['model']['model_path'][-1]['before_com_x'] for r in rows if r['active']]),indent=2))
if __name__=='__main__':main()
