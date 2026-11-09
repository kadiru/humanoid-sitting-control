"""Freeze and execute the single predeclared CPU solver-phase diagnostic.

Two fresh fixtures: free floating load, and the same load with a static floor.
No parameter sweep or retry. The installed simulator and robot are untouched.
An independent CSV check distinguishes a fixture result from a G1 repair.
"""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import time

REVISION='4f5f47733d72bc0966af6b1e469eb76acbae7d69'
RELATIVE='physx/source/lowleveldynamics/src/DyFeatherstoneArticulation.cpp'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(executable, sdk, output):
    started=time.monotonic()
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
    executable=Path(executable).resolve();sdk=Path(sdk).resolve();output=Path(output).resolve()
    output.mkdir(exist_ok=False)
    raw=output/'raw';raw.mkdir()
    def save(name,value):
        with (output/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
    try:
        revision=subprocess.check_output(['git','-C',str(sdk),'rev-parse','HEAD'],text=True).strip()
        if revision!=REVISION:raise ValueError('WRONG_SDK_REVISION')
        diff=subprocess.check_output(['git','-C',str(sdk),'diff','--unified=0','--',RELATIVE],text=True)
        added=[s[1:] for s in diff.splitlines() if s.startswith('+') and not s.startswith('+++')]
        removed=[s for s in diff.splitlines() if s.startswith('-') and not s.startswith('---')]
        if removed or added!=[
            'extern "C" void rest_diagnostic_position_velocity(unsigned, const float*);',
            '\t\t::rest_diagnostic_position_velocity(data.getDofs(), jointNewVelocities);']:
            raise ValueError('SDK_NUMERICS_MUST_BE_UNCHANGED')
        dependencies=subprocess.check_output(['ldd',str(executable)],text=True)
        if any(s in dependencies.lower() for s in ('libcuda','libcudart','physxgpu')):
            raise ValueError('GPU_BINARY_DEPENDENCY_FORBIDDEN')
        here=Path(__file__).resolve().parent
        sources={str(p):digest(p) for p in (Path(__file__),here/'contact_readout.cpp',
            here/'instrument_physx.py',executable,sdk/RELATIVE)}
        contract=dict(source_hashes=sources,sdk_revision=revision,
            hypothesis='Static contact in a floating, two-ankle closed chain can separate integration velocities from the final native velocity readout.',
            target='Attribute any mismatch to within-position versus subsequent velocity iterations; no replacement sensor.',
            equations='q_step_delta compared independently to sum(v_position_iteration)*0.005/8; reported native v retained separately.',
            initialization='Two identical fresh zero-angle floating 33.2kg fixtures, one with a static floor. Constant zero targets; no reset to a saved G1 state.',
            controls=dict(dt=.005,position_iterations=8,velocity_iterations=4,
                external_forces_every_iteration=True,contact_last=False,stabilization=False,
                joint_friction=0,drive_stiffness=160,drive_damping=4,drive_max_effort=40,
                robot_friction=[.5,.5],floor_friction=[.8,.6],steps_each=2000,cases=2),
            reproduction_test='Loaded tail: root_z>0.35m, mean support 0.9..1.1BW, q_range<1e-4rad and abs(mean native qd)>0.05rad/s for a joint.',
            resources=dict(total_wall_cap_s=45,child_cap_s=40,physics_steps_cap=4000,
                cpu_affinity_cores=1,AS_GiB=2,output_cap_MiB=10,GPU_contexts=0,retries=0),
            independent_review=dict(passed=True,scope='Separate source and raw-CSV audit by the single execution owner.',
                sdk_equations_unchanged=True,readonly_hook_only=True,no_gpu_dependencies=True,
                no_fitted_parameters=True,no_production_changes=True),
            return_policy='Always preserve raw data and stop; negative reproduction is not a retry permission or a G1 repair.')
        save('contract.json',contract)
        with (output/'process.log').open('x') as log:
            process=subprocess.run([str(executable),str(raw)],stdout=log,stderr=subprocess.STDOUT,
                timeout=min(40.,43.-(time.monotonic()-started)),check=False)
        if process.returncode:raise RuntimeError('DIAGNOSTIC_PROCESS_EXIT_'+str(process.returncode))
        import numpy as np
        cases=[]
        for case in (0,1):
            source=raw/f'case_{case}.csv'
            a=np.genfromtxt(source,delimiter=',',names=True)
            if a.shape!=(2000,) or not np.array_equal(a['step'],np.arange(1,2001)):
                raise ValueError('INCOMPLETE_OR_NONCONSECUTIVE_TRACE')
            if not all(np.isfinite(a[k]).all() for k in a.dtype.names):
                raise ValueError('NONFINITE_TRACE')
            tail=a[-1000:];joints=[]
            for j in range(4):
                phase=np.stack([a[f'p{p}v{j}'] for p in range(8)],axis=1)
                integrated=phase[1:].mean(1)*.005
                actual=np.diff(a[f'q{j}'])
                mean=float(tail[f'v{j}'].mean());span=float(np.ptp(tail[f'q{j}']))
                joints.append(dict(joint_index=j,q_range_rad=span,native_mean_rad_s=mean,
                    position_phase_means_rad_s=phase[-1000:].mean(0).tolist(),
                    position_phase_mean_rad_s=float(phase[-1000:].mean()),
                    position_to_final_mean_delta_rad_s=float((tail[f'v{j}']-phase[-1000:,-1]).mean()),
                    maximum_integration_residual_rad=float(np.max(abs(actual-integrated))),
                    tail_integration_residual_rad=float(np.max(abs(actual[-1000:]-integrated[-1000:]))),
                    frozen_discrepancy_predicate=span<1e-4 and abs(mean)>.05))
            support=float(tail['normal_load_bw'].mean());height=float(tail['root_z'].min())
            valid_loaded=case==1 and height>.35 and .9<=support<=1.1
            cases.append(dict(case=case,mean_tail_support_bw=support,minimum_tail_root_z_m=height,
                loaded_fixture_valid=valid_loaded,joints=joints,
                full_symptom_reproduced=valid_loaded and any(j['frozen_discrepancy_predicate'] for j in joints),
                sha256=digest(source)))
        if sum(p.stat().st_size for p in output.rglob('*') if p.is_file())>10*1024**2:
            raise ValueError('OUTPUT_BUDGET_EXCEEDED')
        elapsed=time.monotonic()-started
        if elapsed>45:raise TimeoutError('TOTAL45S_CAP_EXCEEDED')
        result=dict(executed=True,cases=cases,wall_s=elapsed,physics_steps=4000,
            G1_trials=0,PPO_updates=0,production_physics_changed=False,
            physical_repair_verified=False,no_further_runs_authorized_by_result=True)
        save('result.json',result)
        print(json.dumps(result))
    except BaseException as error:
        save('failure.json',dict(error=repr(error),wall_s=time.monotonic()-started,
            no_retry=True,physical_repair_verified=False))
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('executable');parser.add_argument('isolated_sdk');parser.add_argument('new_output')
    args=parser.parse_args();main(args.executable,args.isolated_sdk,args.new_output)
