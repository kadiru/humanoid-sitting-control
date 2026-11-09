"""Independent synthetic-measurement contract tests; no simulated measurements.

Synthetic planes below are unit fixtures only. They are never installed in a
runtime camera/observation. Check exact model geometry under each path pose and
explicitly reject unavailable/current-plane and inconsistent-foot conditions.
"""
from pathlib import Path
import datetime,json,hashlib
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel
from low_seat_measured_guard_130 import MeasuredGuard
from low_seat_gated_command_128 import support_limit
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'measured_guard_audit_131'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    O.mkdir(exist_ok=False);m=BodyModel(D/'cooked_model_50/model.json')
    tr=np.load(D/'normal_landing3s_06/candidate/trace.npz');at=int(np.argmin(abs(tr['t']-1)))
    rows=[];errors=[];rejections=[]
    for c in [0,1,6,7]:
        path=read(D/'fixed_runtime_preparation_127'/f'plan_{c}.json');geo=tr['geometry'][at,c].copy()
        guard=MeasuredGuard(m,path,geo)
        initial=np.array(path['path'][0]['root']);R0=Rotation.from_quat(initial[3:]).as_matrix()
        raw_n=R0@geo[2:5];raw_n/=np.linalg.norm(raw_n);raw_d=float(geo[5]-raw_n@initial[:3])
        for k,e in enumerate(path['path']):
            pose=np.array(e['root']);R=Rotation.from_quat(pose[3:]).as_matrix()
            synthetic=geo.astype(float);synthetic[2:5]=R.T@raw_n;synthetic[5]=raw_d+raw_n@pose[:3];synthetic[11:]=[1.,0.]
            q=np.array(e['q']);g=-R[2,:]
            a=guard.observe(q,g,synthetic);assert a['available'] and a['anchor_valid'],a
            errors.append(max(abs(a['housing_clearance_m']-e['clearance_m']),abs(a['pelvis_gap_m']-e['pelvis_gap_m']),abs(a['patch_margin_m']-e['patch_margin_m']),abs(a['plane_height_error_m'])))
            assert errors[-1]<1e-8,(c,k,errors[-1],a)
            assert np.max(abs(np.array(a['root_estimate'])-pose[:3]))<1e-8
            rows.append(dict(case=c,knot=k,maximum_geometry_error_m=errors[-1],anchor_valid=True))
        tests=[('missing',dict(index=11,value=0.)),('stale',dict(index=12,value=.501)),('nan',dict(index=5,value=float('nan')))]
        for name,t in tests:
            changed=synthetic.copy();changed[t['index']]=t['value'];r=guard.observe(q,g,changed)
            assert not r['available'];rejections.append(dict(case=c,test=name,rejected=True))
        qbad=q.copy();qbad[m.names.index('right_ankle_pitch_joint')]+=.02
        r=guard.observe(qbad,g,synthetic);assert r['available'] and not r['anchor_valid']
        rejections.append(dict(case=c,test='inconsistent_foot_rotation',rejected=True))
        # A new measured plane displacement remains observable; no replay/cache
        # is used to make the old plane appear current.
        changed=synthetic.copy();changed[5]-=.01
        r=guard.observe(q,g,changed);assert r['available'] and abs(r['plane_height_error_m'])>.009
        rejections.append(dict(case=c,test='new_current_plane_not_replaced',rejected=True))
    files=[Path(__file__).resolve(),Path(__file__).with_name('low_seat_measured_guard_130.py').resolve(),D/'cooked_model_50/model.json']+[D/'fixed_runtime_preparation_127'/f'plan_{c}.json' for c in [0,1,6,7]]
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,
             scope='364 exact synthetic unit-fixture plane/pose checks and20 rejection cases. NOT physical sensor replay, live rendering, estimator accuracy validation or sensed support. Synthetic inputs exist only in this audit. Runtime guard consumes current measured plane and encoders/IMU with explicit fixed-foot assumption.',
             records=len(rows),max_geometry_replay_error_m=max(errors),rows=rows,rejections=rejections,
             physics_ticks=0,optimizer_updates=0,holdout_used=False,physical_admission=False,
             source_sha256={str(p):sha(p) for p in files})
    (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ['rows','rejections','source_sha256']}),flush=True)
if __name__=='__main__':main()
