"""Independent force/command audit of144 and pure current-input implementation145.

Exact preserved geometry is tied to audited134, while all364 force and command
records are recomputed by finite-difference virtual work. Zero physics.
"""
from pathlib import Path
import copy,datetime,hashlib,json
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from low_seat_balanced_force_145 import reallocate
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    O=D/'balanced_force_audit_146';O.mkdir(exist_ok=False)
    source=D/'fixed_foot_probe_134/candidate/fixed_plans_132.json';originals=read(source)
    oldaudit=read(D/'fixed_foot_probe_134/independent_fixed_audit_135.json')
    assert oldaudit['software_audit_passed'] and not oldaudit['physical_admission_passed']
    revision=read(D/'balanced_force_revision_144.json')
    for p,h in revision['source_sha256'].items():assert sha(p)==h,p
    mp=D/'cooked_model_50/model.json';ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json'
    m=BodyModel(mp);ine=read(ip);mass=np.array(ine['mass']);W=mass.sum()*9.81
    limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'}
    kp=np.array([200,150,150,200,160,160]*2+[40]*4);eff=np.array([300,300,300,300,20,20]*2+[300]*4)
    changed={'load_checks','min_command_margin_rad','maximum_effort_utilization','all_endpoint_model_checks_passed'}
    maxwork=0.;maxcmd=0.;maxeq=0.;rows=[];files=[Path(__file__).resolve(),Path(__file__).with_name('low_seat_balanced_force_145.py'),mp,ip,source,D/'balanced_force_revision_144.json']
    for original in originals:
        c=original['case'];p=D/f'balanced_force_paths_144/plan_{c}.json';files.append(p);saved=read(p);plan=saved['plan'];old=original['plan'];inputs=original['inputs']
        assert saved['inputs']==inputs
        replay=reallocate(m,ine,old,inputs);assert replay['summary']['case']==-1
        replay['summary']['case']=c # Pure runtime has no case labels; report-only field.
        assert replay==plan,('pure_replay',c)
        for key in old:
            if key not in ['path','summary','model_passed']:assert plan[key]==old[key],(c,key)
        assert len(plan['path'])==len(old['path'])==91
        ids=plan['moving_ids'];fixed=[i for i in range(37) if i not in ids];q0=np.array(inputs['q']);cmd0=np.array(inputs['command']);tau0=np.array(plan['initial_static_torque_Nm'])
        n=np.array(plan['normal']);minimum_cop=1.;casework=0.
        for k,(e,prior) in enumerate(zip(plan['path'],old['path'])):
            for key,value in prior.items():
                if key not in changed:assert e[key]==value,(c,k,key)
            q=np.array(e['q']);root=np.array(e['root']);T=transform(Rotation.from_quat(root[3:]).as_matrix(),root[:3]);fk=m.fk(q)
            load=e['load_checks'][0];oldload=prior['load_checks'][0]
            for key in ['normal_load_fraction','vertical_seat_fraction','seat_contact_point','seat_force_normalized','corner_world_points','corner_sides']:
                assert load[key]==oldload[key],(c,k,key)
            P=np.array(load['corner_world_points']);w=np.array(load['corner_fractions']);sides=np.array(load['corner_sides']);sf=np.array(load['seat_force_normalized']);contact=np.array(load['seat_contact_point'])
            C=np.average([apply(T@fk[b],cc) for b,cc in zip(ine['body_names'],ine['local_com'])],axis=0,weights=mass)
            assert max(abs(C-e['com']))<1e-12
            resultant=np.r_[0.,0.,w.sum()]+sf-np.array([0.,0.,1.])
            moment=np.sum(np.cross(P-C,np.c_[np.zeros((len(w),2)),w]),axis=0)+np.cross(contact-C,sf)
            equilibrium=float(max(np.max(abs(resultant)),np.max(abs(moment))));maxeq=max(maxeq,equilibrium)
            assert equilibrium<1e-8 and min(w)>=-1e-10 and abs(sf@n-load['normal_load_fraction'])<1e-12
            if e['pelvis_gap_m']>1e-8:assert np.array_equal(sf,np.zeros(3))
            local=[]
            for point,side in zip(P,sides):local.append(np.linalg.inv(T@fk[side+'_ankle_roll_link'])@np.r_[point,1.])
            for side in ['left','right']:
                take=sides==side;force=w[take].sum();assert force>0
                cop=np.sum(P[take,:2]*w[take,None],axis=0)/force;H=ConvexHull(P[take,:2]).equations
                margin=float(min(-(H[:,:2]@cop+H[:,2])));minimum_cop=min(minimum_cop,margin)
                assert margin>=.005-1e-10 and abs(margin-load['individual_foot_cop'][side]['margin_m'])<1e-12
            tau=[]
            for j in ids:
                values=[]
                for sign in [-1,1]:
                    qq=q.copy();qq[j]+=sign*1e-5;f=m.fk(qq)
                    gravity=sum(mm*9.81*apply(T@f[b],cc)[2] for b,mm,cc in zip(ine['body_names'],mass,ine['local_com']))
                    work=sum(W*ww*(T@f[s+'_ankle_roll_link']@ll)[2] for ww,s,ll in zip(w,sides,local))
                    values.append(gravity-work)
                tau.append((values[1]-values[0])/2e-5)
            tau=np.array(tau);err=float(max(abs(tau-load['static_torque_Nm'])));assert err<1e-6,(c,k,err)
            maxwork=max(maxwork,err);casework=max(casework,err)
            cmd=np.array(load['command']);expected=cmd0.copy();expected[ids]+=q[ids]-q0[ids]+(tau-tau0)/kp
            ce=float(max(abs(cmd-expected)));assert ce<2e-8,(c,k,ce);maxcmd=max(maxcmd,ce)
            assert np.array_equal(cmd[fixed],cmd0[fixed]) and np.array_equal(q[fixed],q0[fixed])
            cm=min(min(cmd[j]-limits[m.names[j]][0],limits[m.names[j]][1]-cmd[j]) for j in ids)
            eu=max(abs(np.array(load['static_torque_Nm']))/eff)
            assert cm>=.03 and eu<=1 and abs(cm-load['min_command_margin_rad'])<1e-12 and abs(eu-load['maximum_effort_utilization'])<1e-12
            assert load['passed'] and e['model_pose_passed'] and e['all_endpoint_model_checks_passed']
        assert np.array_equal(plan['path'][0]['load_checks'][0]['command'],cmd0)
        assert plan['path'][0]['load_checks'][0]['corner_fractions']==old['path'][0]['load_checks'][0]['corner_fractions']
        command=np.array([e['load_checks'][0]['command'] for e in plan['path']]);duration=np.max(abs(np.diff(command,axis=0)),axis=1).sum()/.15
        assert abs(duration-plan['summary']['command_polyline_time_s'])<1e-12
        rows.append(dict(case=c,knots=91,pure_current_input_replay_exact=True,all_geometry_records_exact=True,minimum_individual_foot_cop_margin_m=minimum_cop,max_virtual_work_error_Nm=casework,static_command_polyline_s=float(duration),physical_admission=False));print(json.dumps(rows[-1]),flush=True)
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,records=364,scope=__doc__,checks=rows,max_virtual_work_error_Nm=maxwork,max_command_replay_error_rad=maxcmd,max_equilibrium_residual=maxeq,original134_physical_and_exact_gates_remain_failed=True,report_case_field_excluded_from_pure_function=True,physics_ticks=0,optimizer_updates=0,holdout_used=False,physical_admission=False,source_sha256={str(p):sha(p) for p in files})
    (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()
