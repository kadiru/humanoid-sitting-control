"""Independent diagnostic and command-only balance-controller checks.

Synthetic kinematic tracking/perturbations are software fixtures only. They
never enter a simulator and do not predict measured contact or stability.
"""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from low_seat_gated_command_128 import GatedCommandPath
from low_seat_balance_feedback_160 import BalanceFeedback
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'balance_feedback_audit_162'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    O.mkdir(exist_ok=False);torch.set_num_threads(2)
    model=BodyModel(D/'cooked_model_50/model.json');inertia=read(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
    mass=np.array(inertia['mass']);W=mass.sum()*9.81
    diagnostic=read(D/'dynamic_balance_diagnosis_159/results.json');diagnostic_checks=[]
    # Independent FK sum from only the recorded pelvis and joint coordinates;
    # diagnostic159 instead transformed every recorded body separately.
    for variant in ['normal_landing3s_06','balanced_force_probe_150']:
        P=D/variant/'candidate';tr=dict(np.load(P/'trace.npz'));res=read(P/'results.json');origin=np.array(res['origins']);bi=res['body_names'].index('pelvis')
        for row in [r for r in diagnostic['rows'] if r['variant']==variant]:
            c=row['case']
            for k in [9,19,29,49,64,69,74,99,149,199]:
                b=tr['body'][k,c,bi];R=Rotation.from_quat(b[3:]).as_matrix();fk=model.fk(tr['q'][k,c])
                C=b[:3]-origin[c]+R@np.average([apply(fk[n],v) for n,v in zip(inertia['body_names'],inertia['local_com'])],axis=0,weights=mass)
                err=float(max(abs(C-np.array(row['series'][k]['actual_com_m']))));assert err<1e-5,(variant,c,k,err)
                diagnostic_checks.append(err)
    limits={j['name']:np.deg2rad(j['limits']) for j in model.joints if j['type']=='PhysicsRevoluteJoint'}
    lo=torch.tensor([limits[n][0]+.03 for n in model.names[:23]],dtype=torch.float32);hi=torch.tensor([limits[n][1]-.03 for n in model.names[:23]],dtype=torch.float32)
    tests=[]
    for case in [0,1,6,7]:
        plan=read(D/f'balanced_force_paths_144/plan_{case}.json')['plan']
        for mode in ['nominal','pitch_disturbance','paused_disturbance','no_support']:
            e=GatedCommandPath(plan,model);e.cap=np.float32(.00225);f=BalanceFeedback(model,plan)
            roots=np.array([r['root'] for r in plan['path']]);Q=np.array([r['q'] for r in plan['path']]);angles=Rotation.from_quat(roots[:,3:]).as_euler('xyz')
            targets=[];offsets=[];positions=[];records=[];reference_goal=f.pitch[0]
            for tick in range(1000):
                progress=not(mode=='paused_disturbance' and 100<=tick<150);allowed=.0 if mode=='no_support' else .1
                nominal=e.advance(progress,allowed)
                q=np.array([np.interp(e.position,e.arc,Q[:,j]) for j in range(37)])
                abc=np.array([np.interp(e.position,e.arc,angles[:,j]) for j in range(3)])
                goal=abc[1];rate=(goal-reference_goal)/.02;reference_goal=goal
                disturbance=(-.05*np.sin(np.pi*min(max((tick-25)/100,0),1))) if mode in ['pitch_disturbance','paused_disturbance'] else 0.
                abc[1]+=disturbance;R=Rotation.from_euler('xyz',abc).as_matrix();g=-R.T[:,2]
                wrench=np.array([[0,W/2,0],[0,W/2,0]]);gyro=np.array([0.,rate,0.])
                target,r=f.update(q,g,gyro,e.position,nominal,wrench,[20.,20.],W)
                # Recompute scalar feedback separately from the returned fields.
                RR=level_rotation(g);fk=model.fk(q);foot_error=[];xs=[]
                for side in ['left','right']:
                    a=plan['path'][0]['foot_targets'][side];F=fk[side+'_ankle_roll_link']
                    xs.append(np.array(a['position'])-RR@F[:3,3]);foot_error.append(np.linalg.norm(Rotation.from_matrix(np.array(a['rotation']).T@RR@F[:3,:3]).as_rotvec()))
                raw=.6*(np.arctan2(g[0],-g[2])-goal)+.15*(gyro[1]-rate)
                if max(foot_error)<=np.deg2rad(.5):raw+=float(np.mean(xs,axis=0)[0]-np.interp(e.position,e.arc,roots[:,0]))
                assert abs(raw-r['raw_correction_rad'])<1e-12
                assert max(abs(f.offset))<=.06+1e-7
                assert np.array_equal(target[23:],nominal[23:]);assert np.max(abs(np.array(r['offset_after'])-np.array(r['offset_before'])))<=.00075+1e-7
                positions.append(e.position);targets.append(target);offsets.append(f.offset.copy());records.append(r)
            if mode=='no_support':assert e.position==e.arc[80]
            complete=next((k+1 for k,s in enumerate(positions) if s==e.arc[-1]),None)
            if mode=='nominal':assert complete is not None and complete*.02<=13.5
            results=[];clipped=[]
            for device in ['cpu','cuda:0']:
                command=torch.tensor(e.Q[0],device=device,dtype=torch.float32);arr=[];count=0
                for target in targets:
                    desired=torch.tensor(target[:23],device=device);count+=int(((desired<lo.to(device))|(desired>hi.to(device))).sum())
                    command[:23]+=(.1*(desired.clamp(min=lo.to(device),max=hi.to(device))-command[:23])).clamp(-.003,.003)
                    arr.append(command.cpu().numpy().copy())
                results.append(np.array(arr));clipped.append(count)
            assert np.array_equal(results[0],results[1]);assert not any(clipped),clipped
            arr=results[0];before=np.vstack([e.Q[0].astype(np.float32),arr[:-1]])
            assert np.all(arr<=before+np.float32(.003)) and np.all(arr>=before-np.float32(.003))
            assert np.array_equal(arr[:,23:],np.broadcast_to(e.Q[0,23:].astype(np.float32),arr[:,23:].shape))
            result=dict(case=case,mode=mode,command_ticks=1000,cpu_cuda_exact=True,clipped_targets=0,max_offset_rad=float(np.max(abs(offsets))),
                maximum_applied_step_rad=float(np.max(abs(arr-before))),nominal_complete_s=complete*.02 if complete else None,
                model_load_permission_synthetic=True,physical_prediction=False)
            tests.append(result);print(json.dumps(result),flush=True)
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,tests=tests,independent_com_replays=len(diagnostic_checks),
        max_independent_com_error_m=max(diagnostic_checks),physics_ticks=0,optimizer_updates=0,physical_admission=False,
        source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),Path(__file__).with_name('low_seat_balance_feedback_160.py'),D/'dynamic_balance_diagnosis_159/results.json',D/'balanced_force_paths_144/SHA256.json']})
    (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()
