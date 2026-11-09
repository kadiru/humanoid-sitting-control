"""Check a saved path through the original Torch actuator filter on the CPU.

All sensor inputs here are synthetic. Passing demonstrates sequencing and
actuator compatibility only; it does not admit a path or certify a robot trial.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
import numpy as np
from cycle_executor import CyclePlan, CycleExecutor, Measurements


def check(plan_path, metadata_path, policy_directory):
    started = time.monotonic()
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    policy_directory = Path(policy_directory).resolve()
    sys.path.insert(0, str(policy_directory))
    spec = importlib.util.spec_from_file_location('original_policy', policy_directory/'policy.py')
    policy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(policy)
    source = json.loads(Path(plan_path).read_text())
    metadata = json.loads(Path(metadata_path).read_text())
    plan = CyclePlan(source, metadata['lower'][:23], metadata['upper'][:23],
                     hover_node=16, touch_node=20, loaded_node=24, release_node=24)
    # These flags permit this synthetic protocol test only. They are not
    # written back to the plan, runtime configuration, or admission records.
    executor = CycleExecutor(plan, current_path_admitted=True, loaded_observer_admitted=True)
    previous = torch.tensor(plan.initial, dtype=torch.float32)
    lower = torch.tensor(plan.lower, dtype=torch.float32)
    upper = torch.tensor(plan.upper, dtype=torch.float32)
    counts = {'admitted_forward_segment': 0, 'admitted_return_segment': 0}
    max_step = max_command_error = 0.
    stage_samples = {}
    returned_at = None
    for control in range(2001):
        index = executor.index
        if executor.phase == 'sit':
            pitch = {0: plan.standing_pitch,
                plan.hover.command_count: plan.hover.pitch,
                plan.touch.command_count: plan.touch.pitch,
                plan.loaded.command_count: plan.loaded.pitch,
                len(plan.forward): plan.rest_pitch}.get(index, plan.standing_pitch)
            loaded = index >= plan.loaded.command_count
            separated = index == 0
        else:
            pitch = plan.standing_pitch if index == len(plan.backward) else plan.release.pitch
            separated = index >= plan.release.command_count
            loaded = not separated
        sample = Measurements(control*4, (0.,)*23, (math.sin(pitch), 0., -math.cos(pitch)),
            0., True, 0., True, True, .38 if loaded else 0., .42 if loaded else 0., separated)
        desired, result = executor.command(sample, previous.numpy())
        if result['aborted']:
            raise AssertionError(result)
        applied = previous.clone()
        applied[:23] = policy.command(previous[:23], torch.tensor(desired), lower, upper)
        max_step = max(max_step, float((applied-previous).abs().max()))
        if max_step > .003+2e-7:
            raise AssertionError('ORIGINAL_ACTION_LIMIT_CHANGED')
        if not torch.equal(applied[23:], torch.tensor(plan.initial[23:], dtype=torch.float32)):
            raise AssertionError('FINGERS_MOVED')
        reason = result['reason']
        stage_samples[reason] = stage_samples.get(reason, 0)+1
        if reason in counts:
            key = 'forward' if reason == 'admitted_forward_segment' else 'backward'
            expected = np.asarray(source[key]['commands'][counts[reason]], np.float32)
            max_command_error = max(max_command_error, float(np.max(abs(applied.numpy()-expected))))
            if max_command_error > 2e-7:
                joint = int(np.argmax(abs(applied.numpy()-expected)))
                raise AssertionError(dict(reason='RECORDED_FLOAT32_ALLOWANCE_EXCEEDED',
                    direction=key, command=counts[reason], joint=joint,
                    actual=float(applied[joint]), expected=float(expected[joint]),
                    previous=float(previous[joint]), desired=float(desired[joint]),
                    error=max_command_error))
            counts[reason] += 1
        previous = applied
        if result['phase'] == 'return' and returned_at is None:
            returned_at = control
            if returned_at != 1000 or result['hold_samples'] < 250:
                raise AssertionError('ORIGINAL_HOLD_WINDOW_CHANGED')
        if result['sensor_cycle_complete']:
            break
    else:
        raise AssertionError('NO_FINITE_SENSOR_PROTOCOL_COMPLETION')
    assert counts == {'admitted_forward_segment': len(plan.forward),
                      'admitted_return_segment': len(plan.backward)}
    np.testing.assert_allclose(previous.numpy(), plan.initial, rtol=0, atol=2e-7)
    assert result['quiet_samples'] == 100 and not result['physical_success']
    files = [Path(plan_path), Path(metadata_path), policy_directory/'policy.py',
             policy_directory/'common.py', Path(__file__), Path(__file__).with_name('cycle_executor.py'),
             Path(__file__).with_name('contact_gate.py')]
    return dict(passed=True, synthetic_measurements=True, G1_trials=0, PPO_updates=0,
        physical_success=False, path_admission_changed=False, original_filter='Torch CPU float32',
        numpy_version=np.__version__,torch_version=torch.__version__,
        command_counts=counts, stage_samples=stage_samples,
        return_start_control=returned_at, completion_control=control,
        synthetic_cycle_time_s=control*.02, maximum_command_difference_rad=max_command_error,
        maximum_applied_step_rad=max_step, actual_wall_s=time.monotonic()-started,
        sources={str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('plan'); parser.add_argument('metadata'); parser.add_argument('policy_directory')
    args = parser.parse_args()
    print(json.dumps(check(args.plan, args.metadata, args.policy_directory), indent=2))
