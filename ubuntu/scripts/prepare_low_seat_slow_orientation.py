"""Offline development-only path checks before the bounded physical probe."""
import json,copy,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel
import low_seat_normal_timing_controller as normal
import low_seat_slow_orientation_controller as changed
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def main():
    p=D/'normal_landing3s_06/candidate';read=lambda n:json.loads((p/n).read_text())
    model=BodyModel('runs/g1_kinematics_geometry_04/model.json');inertial=read('static_inertial_model.json')
    plans=read('rolling_plan_audit.json');paths=np.load(p/'last_rolling_paths.npz');t=np.load(p/'trace.npz')
    cfg=json.loads((D/'normal_landing3s_06.json').read_text());idx=int(np.argmin(abs(t['t']-10.)))
    ids=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    rows=[]
    for a in plans:
        i=a['case'];q=t['q'][idx,i];command=t['command'][idx,i]
        R=Rotation.from_quat(t['body'][idx,i,0,3:]).as_matrix();gravity=R.T@np.array([0.,0.,-1.])
        # Saved model/measurement inputs only. The original planner output is
        # reused for an offline comparison, not claimed as a new runtime test.
        normal.previous=lambda *args,**kw:(paths['command'][i].copy(),paths['q'][i].copy(),copy.deepcopy(a))
        cs,qs,b=changed.plan(model,q,gravity,None,None,command,None,inertial=inertial,config=cfg)
        assert np.isfinite(cs).all() and np.isfinite(qs).all() and b['ik_error']<.003
        assert np.array_equal(cs[0],command)
        rate=np.max(abs(np.diff(cs[:,ids],axis=0)),axis=0)*24/5
        entry=dict(case=i,active=b['slow_orientation_probe']['applied'],max_leg_command_rate_rad_s=rate.tolist(),
            ik_error=b['ik_error'],min_model_housing_clearance_m=min(b['model_clearance_path']),
            endpoint_clearance_m=b['model_clearance_path'][-1],
            min_model_pelvis_patch_margin_m=min(b['model_patch_margin_path']),
            first_command_exact=True)
        rows.append(entry)
    out=D/'slow_orientation_screen_08.json';assert not out.exists()
    out.write_text(json.dumps(dict(scope='Offline only. Inherited applied command slew remains0.15rad/s; target rates above that imply tracking lag. No physical admission implied.',rows=rows),indent=2))
    for r in rows[::2]:print(r)
if __name__=='__main__':main()
