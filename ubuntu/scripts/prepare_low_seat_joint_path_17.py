"""Preflight source checks and model screens only; no physical simulation."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel
import low_seat_joint_path_controller_17 as repair
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');ROOT=Path('/home/kadir/humanoid-sitting')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    s=ROOT/'scripts';transforms=[]
    pairs=[('low_seat_contact_controller.py','low_seat_contact_duration_17.py',[
        ('elapsed / 5 * 24','elapsed / float(getattr(self, "repair_duration", [5.]*self.num_envs)[i]) * 24'),
        ('*24/5*(u<24)','*24/float(getattr(self, "repair_duration", [5.]*self.num_envs)[i])*(u<24)')]),
        ('low_seat_load_controller.py','low_seat_load_duration_17.py',[
        ('import low_seat_contact_controller as landing','import low_seat_contact_duration_17 as landing'),
        ('elapsed/5*24','elapsed/float(getattr(self, "repair_duration", [5.]*self.num_envs)[i])*24')]),
        ('low_seat_normal_timing_controller.py','low_seat_normal_duration_17.py',[
        ('import low_seat_load_controller as load','import low_seat_load_duration_17 as load')])]
    for old,new,replacements in pairs:
        before=(s/old).read_text();after=before
        for a,b in replacements:assert after.count(a)==1;after=after.replace(a,b)
        assert after==(s/new).read_text()
        transforms.append(dict(old=old,new=new,old_sha256=sha(s/old),new_sha256=sha(s/new),replacements=replacements))
    p=D/'normal_landing3s_06/candidate';t=np.load(p/'trace.npz');ti=int(np.argmin(abs(t['t']-8.)))
    m=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json');inertia=json.loads((p/'static_inertial_model.json').read_text())
    cal=json.loads((D.parent/'contact_effort_repair_01/plane_calibration.json').read_text());cfg=json.loads((D/'normal_landing3s_guarded_09.json').read_text())
    rows=[]
    for i in range(24):
        q=t['q'][ti,i];g=Rotation.from_quat(t['body'][ti,i,0,3:]).inv().apply([0,0,-1.]);geometry=t['geometry'][ti,i];reference=np.array([t['estimated_root_x'][ti,i],0.,0.])
        plane=repair.measured(m,q,g,geometry,reference,cal);active=repair.eligible(plane)
        row=dict(case=i,selected=active,measured_height_m=plane[3],measured_slope_deg=plane[4])
        if active:
            c,qs,a=repair.plan(m,q,g,geometry,reference,t['command'][ti,i],cal,inertia,cfg)
            assert np.array_equal(c[0],t['command'][ti,i]);assert np.isfinite(c).all() and np.isfinite(qs).all()
            leg=[m.names.index(side+'_'+j+'_joint') for side in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
            other=[j for j in range(37) if j not in leg]
            assert np.array_equal(c[:,other],np.broadcast_to(c[0,other],c[:,other].shape))
            row.update(screen=a['joint_path_17']['screen'],first_command_exact=True,nonleg_reference_fixed=True,
                resampled_nominal_rate_rad_s=float(abs(np.diff(c,axis=0)).max()*24/7))
        rows.append(row)
    assert [r['case'] for r in rows if r['selected']]==[0,1,6,7,10,11]
    out=D/'joint_path_preparation_17.json';assert not out.exists()
    out.write_text(json.dumps(dict(passed=True,scope='Six targeted development trials pass nominal model checks; eight-second prefix explicitly replaces ten-second prefix ONLY for these selected approaches. All physical outcomes pending; 36cm/+20deg not repaired.',
        source_transforms=transforms,rows=rows,physical_trials=0,optimizer_updates=0,fresh_holdout=False,
        inputs={str(p/n):sha(p/n) for n in ['trace.npz','static_inertial_model.json']},source_sha256=sha(__file__)),indent=2))
    print(json.dumps(dict(passed=True,active=6,max_rate=max(r.get('resampled_nominal_rate_rad_s',0.) for r in rows))))
if __name__=='__main__':main()
