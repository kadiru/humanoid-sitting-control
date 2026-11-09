"""Independent Torch CPU/CUDA arithmetic and gating checks, zero physics."""
from pathlib import Path
import datetime,hashlib,json
import numpy as np
import torch
from humanoid_sitting.shared_frame import BodyModel
from posture_episode_ppo import PosturePPO
from low_seat_gated_command_128 import GatedCommandPath,support_limit

D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'gated_command_audit_v2_129'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    O.mkdir(exist_ok=False);m=BodyModel(D/'cooked_model_50/model.json')
    prep=D/'fixed_runtime_preparation_127';pr=read(prep/'results.json');assert pr['all_selected_paths_passed']
    for p,h in pr['source_sha256'].items():assert sha(p)==h,p
    base=D/'normal_landing3s_06/candidate';manifest=read(base/'manifest.json');checkpoint=Path(manifest['args']['checkpoint'])
    assert sha(checkpoint)==manifest['parent_checkpoint_sha256']
    fixture=np.load(base/'sensor_inputs.npz');obs=fixture['parent_observation'][0]
    parent=PosturePPO(np.zeros(414),np.ones(414),0,'cuda:0');parent.load(checkpoint)
    with torch.no_grad():action=parent.choose(torch.tensor(obs,device='cuda:0'),False)[0]
    assert action.shape==(24,3) and torch.isfinite(action).all()
    recorded=fixture['parent_action'][0]
    assert np.array_equal(action.cpu().numpy(),recorded), 'same-device parent fixture replay'
    offsets=(.008*torch.tanh(action)).cpu().numpy()
    body=np.arange(23);six=[m.names.index(s+'_'+j+'_joint') for j in ['hip_pitch','knee','ankle_pitch'] for s in ['left','right']]
    limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'}
    lo=torch.tensor([limits[m.names[j]][0]+.03 for j in body],dtype=torch.float32)
    hi=torch.tensor([limits[m.names[j]][1]-.03 for j in body],dtype=torch.float32)
    tests=[];gate_tests=[]
    good=dict(geometry_valid=True,geometry_age=0.,plane_height_error=0.,plane_slope_error_deg=0.,
              foot_conditions=[20.,20.],foot_wrench=np.array([[0,40,0],[0,40,0]]),weight=100.,normal_z=.96,
              joint_speed=.1,gyro_speed=.05,near_plane=0.,patch_margin=.02,housing_clearance=.0032)
    assert support_limit(**good)['max_normal_bw']==.1
    changes=[('missing',dict(geometry_valid=False)),('stale',dict(geometry_age=.501)),
             ('height_change',dict(plane_height_error=.003001)),('slope_change',dict(plane_slope_error_deg=.50001)),
             ('singular',dict(foot_conditions=[201,20])),('unloaded_foot',dict(foot_wrench=np.array([[0,0,0],[0,80,0]]))),
             ('no_seat_estimate',dict(foot_wrench=np.array([[0,50,0],[0,50,0]]))),
             ('motion',dict(joint_speed=.501)),('gyro',dict(gyro_speed=.201)),
             ('positive_gap',dict(near_plane=.00301)),('patch',dict(patch_margin=.01499)),
             ('housing',dict(housing_clearance=.00299)),('nonfinite',dict(geometry_age=float('nan')))]
    for name,change in changes:
        r=support_limit(**(good|change));assert r['max_normal_bw']==0.,(name,r)
        gate_tests.append(dict(test=name,max_normal_bw=0.,passed=True))
    for c in pr['selected_cases']:
        plan=read(prep/f'plan_{c}.json')
        for mode in ['nominal','no_support','delayed_support']:
            engine=GatedCommandPath(plan,m);targets=[];nominal=[engine.nominal.copy()];positions=[];demands=[]
            for tick in range(1000):
                available=.1 if mode=='nominal' or (mode=='delayed_support' and tick>=750) else 0.
                targets.append(engine.advance(True,available));nominal.append(engine.nominal.copy())
                positions.append(engine.position);demands.append(float(np.interp(engine.position,engine.arc,engine.loads)))
            target=np.array(targets);nominal=np.array(nominal)
            if mode=='no_support':assert max(demands)==0 and engine.position==engine.arc[80]
            if mode=='delayed_support':assert max(demands[:750])==0 and engine.position==engine.arc[-1]
            if mode=='nominal':
                complete=np.flatnonzero(np.array(positions)==engine.arc[-1])[0]+1
                assert complete*.02<=13.5
                if c in [0,6]:
                    original=np.load(D/f'staged_fixed_foot_filtered_{c}_122/command_schedule.npz')
                    assert np.array_equal(target[:complete],original['raw_target'])
                    assert np.array_equal(nominal[:complete+1],original['applied'])
            results=[]
            for device in ['cpu','cuda:0']:
                q=torch.tensor(nominal[0],device=device);tr=[];clipped=0
                lower=lo.to(device);upper=hi.to(device)
                for tick,x in enumerate(target):
                    ref=torch.tensor(x,device=device)
                    # Independent exact composition in inherited command order.
                    off=torch.zeros(37,device=device)
                    if tick>=700: # start1s plus14s => absolute15s frozen-parent handoff
                        for j in range(3):off[six[2*j:2*j+2]]=float(offsets[c,j])
                    desired=ref[body]+off[body]
                    clipped+=int(torch.count_nonzero((desired<lower)|(desired>upper)))
                    q[body]+=torch.clamp(torch.tensor(.1,device=device)*(torch.clamp(desired,min=lower,max=upper)-q[body]),-.003,.003)
                    tr.append(q.cpu().numpy().copy())
                results.append(np.array(tr))
                assert clipped==0,(c,mode,device,clipped)
            assert np.array_equal(results[0],results[1]),(c,mode,'CPU/CUDA mismatch')
            arr=results[1];maxstep=float(np.max(abs(np.diff(np.vstack([nominal[0],arr]),axis=0))))
            previous=np.vstack([nominal[0],arr[:-1]])
            representable_cap_passed=bool(np.all(arr<=previous+np.float32(.003)) and np.all(arr>=previous-np.float32(.003)))
            print(json.dumps(dict(case=c,mode=mode,maxstep=maxstep,original_stricter_delta_test=maxstep<=.003+2e-8,exact_float32_cap=representable_cap_passed)),flush=True)
            assert representable_cap_passed and maxstep<=.003+3e-7 and np.isfinite(arr).all()
            assert np.array_equal(arr[:,23:],np.broadcast_to(nominal[0,23:],arr[:,23:].shape))
            # Before parent handoff, independent actual filter exactly matches nominal.
            assert np.array_equal(arr[:700],nominal[1:701])
            final_delta=arr[-1]-nominal[-1]
            expected=np.zeros(37,np.float32)
            for j in range(3):expected[six[2*j:2*j+2]]=offsets[c,j]
            parent_error=float(np.max(abs(final_delta-expected)))
            assert parent_error<2e-6,(c,mode,parent_error)
            tests.append(dict(case=c,mode=mode,passed=True,cpu_cuda_exact=True,no_target_clipping=True,
                              fixed_14_fingers=True,max_applied_step_rad=maxstep,original_stricter_delta_test=maxstep<=.003+2e-8,exact_float32_cap_passed=representable_cap_passed,existing_subtraction_tolerance_rad=3e-7,parent_not_cancelled_max_error_rad=parent_error,
                              maximum_assumed_load_bw=max(demands),nominal_duration_s=complete*.02 if mode=='nominal' else None))
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,
             scope='Original129 aborted at an unnecessarily stricter2e-8 subtraction test. It remains preserved. This version checks EXACT float32 before-plus/minus-cap bounds and the EXISTING3e-7 delta tolerance from audits31/36; no command/filter/rate/physical criterion changed. Four current-input planned paths; nominal schedule parity, zero/delayed sensor-permission tests, actual frozen414 parent fixture replay and independent Torch CPU/CUDA command composition. 12000command-only ticks per device, ZERO physics. No actual417 runtime or sensed-contact/dynamic/retention claim.',
             tests=tests,guard_tests=gate_tests,parent_sha256=sha(checkpoint),parent_fixture_action_exact=True,
             parent_device=torch.cuda.get_device_name(0),physics_ticks=0,optimizer_updates=0,holdout_used=False,physical_admission=False,
             source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),Path(__file__).with_name('low_seat_gated_command_128.py'),prep/'results.json',checkpoint,base/'sensor_inputs.npz']})
    assert sha(checkpoint)==manifest['parent_checkpoint_sha256']
    (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ['tests','guard_tests','source_sha256']}),flush=True)
if __name__=='__main__':main()
