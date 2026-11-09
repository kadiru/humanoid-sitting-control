"""One bounded check of a previously untested path-interpolation premise.

No optimizer, physics, parameter scan, or counterfactual rollout. Inputs and
equations are frozen before numerical imports; outcomes never authorize motion.
"""
import os
import time
START=time.monotonic()
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import argparse
import hashlib
import json
from pathlib import Path
import resource
import signal
import sys


def main(project,source,output,review_log):
    def deadline(*_):raise TimeoutError('45_SECOND_TOTAL_INTERPOLATION_AUDIT')
    signal.signal(signal.SIGALRM,deadline);signal.setitimer(signal.ITIMER_REAL,max(.001,45-(time.monotonic()-START)))
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3))
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    project=Path(project).resolve();source=Path(source).resolve();output=Path(output).resolve()
    output.mkdir(exist_ok=False)
    def write(name,value):
        with (output/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    try:
        exp=project/'experiments'
        paths=dict(plan=exp/'rest_cycle_measured_support_20261109_v2/construction/plan.json',
            frame=project/'outputs/rest_cycle_narrow_support_20261109_v1/camera_and_wrench_capture/frame_0.npz',
            model=project/'runs/g1_kinematics_geometry_04/model.json',
            colliders=exp/'furniture_perception_repair_20261029_v1/render_colliders.json',
            kinematics_source=project/'sitting_task/src/humanoid_sitting/shared_frame.py',
            interpolation_source=source/'interpolation_audit.py',
            review_test_source=source/'test_interpolation_audit.py',
            review_log=Path(review_log).resolve(),runner=Path(__file__).resolve())
        review=paths['review_log'].read_text()
        if 'Ran 2 tests' not in review or not review.rstrip().endswith('OK'):
            raise ValueError('INDEPENDENT_ANALYTICAL_REGRESSION_REVIEW_REQUIRED')
        hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths.values()}
        write('contract.json',dict(source_hashes=hashes,
            target='The actual forward/backward control counts of the saved narrow-support model path.',
            premise='Its clearance audit interpolates q and translation linearly and rotation by Slerp between knots; endpoint mechanics alone do not certify those poses.',
            equations='For every original scheduled fraction u=j/N, T_foot=T_root(u)*FK(q(u)); compare to the original two foot anchors. Evaluate all pelvis collider heights against the original observed plane.',
            independent_review='Analytical closed-chain fixture: exact endpoint closure with interior displacement, plus a near-stationary passing fixture. Both tests passed before this launch.',
            controls=dict(foot_error_limit=2e-5,pelvis_plane_penetration_limit_m=1e-6,
                max_total_controls=2000,solver_calls=0,simulation_steps=0,retries=0),
            resources=dict(total_wall_cap_s=45,cpu_affinity_cores=1,AS_GiB=8,output_cap_MiB=10,GPU_contexts=0),
            return_policy='Preserve result and stop. A sampled-kinematics failure rejects that model interpolation; it is not a physical failure attribution or permission to rerun the robot.'))
        import numpy as np
        sys.path[:0]=[str(source),str(project/'sitting_task/src')]
        from humanoid_sitting.shared_frame import BodyModel,level_rotation
        from interpolation_audit import audit_interpolation
        plan=json.loads(paths['plan'].read_text());model=BodyModel(paths['model'])
        with np.load(paths['frame']) as f:frame=dict(f)
        level=level_rotation(frame['gravity'].astype(float).copy())
        normal=level@frame['geometry'][2:5];normal/=np.linalg.norm(normal)
        offset=float(frame['geometry'][5]);fk=model.fk(frame['q'].astype(float))
        root=np.eye(4);root[:3,:3]=level
        feet=['left_ankle_roll_link','right_ankle_roll_link'];anchors=[root@fk[b] for b in feet]
        shapes=json.loads(paths['colliders'].read_text())['shapes']
        pelvis=np.concatenate([s['vertices'] for s in shapes if s['body']=='pelvis'])
        results={}
        for direction,node_key in (('forward','nodes'),('backward','return_nodes')):
            results[direction]=audit_interpolation(model,plan[node_key],plan[direction],feet,anchors,pelvis,normal,offset)
        elapsed=time.monotonic()-START
        if elapsed>45:raise TimeoutError('45_SECOND_TOTAL_INTERPOLATION_AUDIT')
        value=dict(passed=all(r['passed'] for r in results.values()),directions=results,
            wall_s=elapsed,solver_calls=0,simulation_steps=0,PPO_updates=0,physical_success=False,
            physical_motion_admitted=False,criteria_changed=False,repeat_authorized=False)
        write('result.json',value)
        if sum(p.stat().st_size for p in output.iterdir())>10*1024**2:raise ValueError('OUTPUT_CAP_EXCEEDED')
        print(json.dumps({**{k:v for k,v in value.items() if k!='directions'},
            'directions':{k:{a:b for a,b in v.items() if a!='rows'} for k,v in results.items()}},indent=2))
    except BaseException as error:
        write('failure.json',dict(error=repr(error),elapsed_s=time.monotonic()-START,no_retry=True,physical_success=False))
        raise
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('project');parser.add_argument('source_directory')
    parser.add_argument('new_output_directory');parser.add_argument('analytical_review_log')
    a=parser.parse_args();main(a.project,a.source_directory,a.new_output_directory,a.analytical_review_log)
