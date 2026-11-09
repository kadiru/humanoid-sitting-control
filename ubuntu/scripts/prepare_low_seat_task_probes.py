"""Validate and launch two bounded pre-training physical research probes."""
from pathlib import Path
import numpy as np,json,hashlib,datetime,subprocess,py_compile
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel
from low_seat_task_controller import task_rhs
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'normal_descent_03/candidate'
m=BodyModel('runs/g1_kinematics_geometry_04/model.json');a=json.loads((P/'rolling_plan_audit.json').read_text());qs=np.load(P/'last_rolling_paths.npz')['q'];checks=[]
for i in [0,2,4,12]:
    # Match the controller's float64 finite-difference calculation.
    q=qs[i,10].astype(float);root=np.array(a[i]['root_path'][10]);R=Rotation.from_quat(root[3:]).as_matrix()
    delta=np.array([.0001,0,.0001]);angle=.0001;Rnew=Rotation.from_euler('y',angle).as_matrix()@R
    for s in ['left','right']:
        body=s+'_ankle_roll_link';fk=m.fk(q);before=root[:3]+R@fk[body][:3,3]
        ids=[m.names.index(s+'_'+j+'_joint') for j in ['hip_pitch','knee','ankle_pitch']];J=[]
        for j in ids:
            v=[]
            for sign in [-1,1]:
                trial=q.copy();trial[j]+=sign*1e-5;foot=m.fk(trial)[body]
                v.append(np.r_[(R@foot[:3,3])[[0,2]],Rotation.from_matrix(R@foot[:3,:3]).as_euler('xyz')[1]])
            J.append((v[1]-v[0])/2e-5)
        shift=np.linalg.solve(np.array(J).T,task_rhs(R@fk[body][:3,3],delta,angle));qnew=q.copy();qnew[ids]+=shift
        after=root[:3]+delta+Rnew@m.fk(qnew)[body][:3,3];error=float(np.linalg.norm(after-before))
        rot_error=float((Rotation.from_matrix(R@fk[body][:3,:3]).inv()*Rotation.from_matrix(Rnew@m.fk(qnew)[body][:3,:3])).magnitude())
        assert error<2e-6 and rot_error<2e-6,(error,rot_error)
        checks.append(dict(case=i,side=s,foot_position_error=error,foot_rotation_error=rot_error))
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
names=['damped_original_01','clearance_damped_01','load_feedback_original_02','load_feedback_clearance_02','normal_descent_03','hover_rotate_normal_03','normal_ankle_balance_04','normal_load_balance_04']
files={str(p):sha(p) for name in names for p in (D/name).rglob('*') if p.is_file() and not p.is_symlink()}
f=D/'rejected_probes_preservation_04.json';assert not f.exists();f.write_text(json.dumps(files,indent=2))
for name,kp,kd in [('normal_task_mild_05',.4,.08),('normal_task_firm_05',.8,.12)]:
    cfg=json.loads((D/'normal_descent_03.json').read_text());cfg.update(task_pitch_kp=kp,task_pitch_kd=kd,task_pitch_cap=.04,task_joint_cap=.08)
    f=D/(name+'.json');assert not f.exists();f.write_text(json.dumps(cfg,indent=2))
sources=[Path('scripts')/n for n in ['prepare_low_seat_task_probes.py','low_seat_task_controller.py','run_low_seat_task_probe.py','launch_low_seat_task_probe.py','low_seat_normal_controller.py','low_seat_load_controller.py','low_seat_contact_controller.py']]
for p in sources:py_compile.compile(str(p),doraise=True)
rev=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),diagnosis='Retiming balance authority alone worsened physical failures. Coordinate translation and pitch through a fixed-foot task Jacobian rather than separate ankle/hip heuristic and translation-only damping.',changed='Selected rows replace inherited bridge balance and old landing correction with bounded six-joint kinematic pose correction. Path, measured geometry, load feedforward, shear feedback, parent and physical criteria unchanged.',hypothesis_only=True,software_fixed_foot_checks=checks,software_initial_test=dict(failed=True,error='Float32 finite-difference test rotation residual2.8386095397520448e-6 exceeded2e-6; translation4.5829501101138484e-7.',repair='Use float64 finite differences, matching the actual controller. No runtime change and no tolerance increase.'),sources={str(p.resolve()):sha(p) for p in sources},budget=dict(variants=2,trials_per_variant=24,active_ticks_per_variant=24000,initial_settling_per_variant=600,optimizer_updates=0),preserved_rejected_files=len(files),fresh_holdout_opened=False)
f=D/'task_revision_05.json';assert not f.exists();f.write_text(json.dumps(rev,indent=2))
(D/'status.json').write_text(json.dumps(dict(phase='coordinated_pose_probes',utc=rev['utc'],main_training_started=False,fresh_holdout_started=False,protected_prior_files=24242,preserved_rejected_files=len(files)),indent=2))
for gpu,name in [(0,'normal_task_mild_05'),(1,'normal_task_firm_05')]:
    command=f'cd /home/kadir/humanoid-sitting && PYTHONPATH=scripts:sitting_task/src IsaacLab/.venv/bin/python scripts/launch_low_seat_task_probe.py --gpu {gpu} --variant {name} > {D}/{name}.log 2>&1'
    subprocess.run(['tmux','new-session','-d','-s',f'low_seat_task_{gpu}',command],check=True)
print('LAUNCHED',rev['utc'],'preserved',len(files),'software_checks',len(checks))
