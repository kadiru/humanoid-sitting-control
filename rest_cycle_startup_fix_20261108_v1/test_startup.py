"""Execute the actual collection/cleanup block with deterministic fake devices.

No IsaacSim import, CUDA context, model solve or physical step. The extracted
AST is the production try/finally block, so name binding and exception paths
are exercised rather than reimplemented. Physics values below are fixtures.
"""
import ast
import contextlib
import json
import signal as deadline_signal
import sys
import tempfile
import traceback
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'rest_cycle_contact_20261108_v1'
sys.path.insert(0, str(OLD))
from recording import write


def collection_block(path):
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    block = next(n for n in ast.walk(main) if isinstance(n, ast.Try)
                 and any(isinstance(x, ast.Call) and isinstance(x.func, ast.Name)
                         and x.func.id == 'save' for x in ast.walk(n)))
    # All fixtures are explicit parameters; assignments in the real block have
    # exactly the same function-local binding behavior as production main().
    names = ['env', 'observer', 'out', 'construction', 'path', 'inertia', 'shapes',
             'SeatPlan', 'audit_geometry', 'audit_commands', 'NormalLanding',
             'save', 'shadow_separation', 'support', 'signals', 'metrics']
    function = ast.FunctionDef(name='run', args=ast.arguments(posonlyargs=[],
        args=[ast.arg(arg=n) for n in names], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=[block], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(np=np, torch=torch, write=write, deadline_signal=deadline_signal,
                     signal=deadline_signal, traceback=traceback)
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['run']


class Device:
    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1
        return 'fake-step'

    def get_physics_step_count(self):
        return self.steps


class View:
    def get_link_velocities(self): return torch.zeros(1, 2, 6)
    def get_masses(self): return torch.ones(1, 2)
    def get_dof_projected_joint_forces(self): return torch.zeros(1, 37)
    def get_dof_actuation_forces(self): return torch.zeros(1, 37)
    def get_contact_force_matrix(self, dt):
        assert dt == .005
        return torch.zeros(1, 1, 3)
    def get_friction_data(self, dt):
        assert dt == .005
        return torch.zeros(32, 3), torch.zeros(32, 3), torch.zeros(1, dtype=torch.int), torch.zeros(1, dtype=torch.int)
    def get_net_contact_forces(self, dt):
        assert dt == .005
        return torch.zeros(2, 3)


def field(shape):
    return types.SimpleNamespace(torch=torch.zeros(*shape))


class Environment:
    def __init__(self, fail_step=None):
        self.device = 'cpu'; self.closed = False; self.fail_step = fail_step
        self.startup = dict(warm_failure=[False], sensor_valid=[True])
        self.model = object(); self.previous = torch.zeros(1, 37)
        self.geometry = np.zeros((1, 13)); self.held = [0]; self.episode_records = []
        data = types.SimpleNamespace(joint_pos=field((1, 37)), joint_vel=field((1, 37)),
            projected_gravity_b=field((1, 3)), root_ang_vel_b=field((1, 3)))
        self.robot = types.SimpleNamespace(data=data, root_view=View())
        sensor = types.SimpleNamespace(contact_view=View())
        self.raw = types.SimpleNamespace(sim=Device(), scene={str(i): sensor for i in range(7)})
        self.raw.scene['all_contacts'] = sensor

    def observation(self): return torch.zeros(1, 4)

    def step(self, command, record):
        assert record
        if self.fail_step is not None and len(self.episode_records) == self.fail_step:
            raise RuntimeError('INJECTED_STEP_FAILURE')
        for _ in range(4): self.raw.sim.step()
        self.episode_records.append(dict(fake_value=np.array([len(self.episode_records)])))

    def save_records(self, path):
        np.savez_compressed(path, **{k: np.stack([r[k] for r in self.episode_records])
                                   for k in self.episode_records[0]})

    def close(self): self.closed = True


class Teacher:
    phase = 'fixture'
    def __init__(self, *args): pass
    def command(self, *args): return np.zeros(23), dict(flag=np.bool_(True))


class StartupTests(unittest.TestCase):
    def run_block(self, directory, env, source=None, geometry=True, commands=True, reset_match=True):
        source = source or HERE / 'trial.py'
        observer = types.SimpleNamespace(last=(None, {'selected_points': np.zeros((3, 3))}))
        support = types.SimpleNamespace(CONTACTS=[(str(i), None, None) for i in range(7)])
        frames = []
        warp = types.SimpleNamespace(to_torch=lambda value: value)
        target = np.zeros(37) if reset_match else np.ones(37)
        with patch.dict(sys.modules, {'warp': warp}), patch.object(deadline_signal, 'alarm') as alarm, \
             patch.object(deadline_signal, 'signal'):
            result = collection_block(source)(env, observer, directory,
                {'nodes': [{'q': target}]}, {}, {}, {},
                lambda *a: object(), lambda *a: {'tests': {'geometry': np.bool_(geometry)}},
                lambda *a: {'passed': np.bool_(commands)}, Teacher,
                lambda *a: frames.append(a[-1]), lambda *a: {'candidate': False}, support, [],
                types.SimpleNamespace(validated=True, frames=0))
        self.assertIn(unittest.mock.call(0), alarm.call_args_list)
        return frames, result

    def test_real_block_completes_and_preserves_substeps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); env = Environment(); original = env.raw.sim.step
            frames, _ = self.run_block(path, env)
            self.assertEqual(frames, [0, 350, 650, 1000])
            self.assertEqual(len(env.episode_records), 1000)
            self.assertTrue(env.closed); self.assertEqual(env.raw.sim.step, original)
            with np.load(path / 'substeps.npz') as data:
                np.testing.assert_array_equal(data['physics_step'], np.arange(1, 4001))
                self.assertEqual(data['velocity'].shape, (4000, 1, 2, 6))
            signals = json.loads((path / 'controller_signals.json').read_text())
            self.assertEqual(len(signals), 1000)
            self.assertTrue(all(x['flag'] is True and x['load_estimator_used'] is False for x in signals))
            audit = json.loads((path / 'fresh_geometry_admission.json').read_text())
            self.assertIs(audit['geometry']['tests']['geometry'], True)

    def test_saved_failure_is_reproduced_in_original_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            env = Environment()
            with self.assertRaises(UnboundLocalError):
                self.run_block(Path(tmp), env, source=OLD / 'trial.py')
            self.assertEqual(env.raw.sim.steps, 0)
            self.assertEqual(len(env.episode_records), 0)

    def test_admission_failures_cannot_move(self):
        for values in [dict(geometry=False), dict(commands=False), dict(reset_match=False)]:
            with self.subTest(values=values), tempfile.TemporaryDirectory() as tmp:
                env = Environment()
                with self.assertRaises(AssertionError): self.run_block(Path(tmp), env, **values)
                self.assertEqual(env.raw.sim.steps, 0); self.assertTrue(env.closed)
                failure = json.loads((Path(tmp) / 'failure.json').read_text())
                self.assertEqual(failure['controls'], 0); self.assertIn('Traceback', failure['traceback'])

    def test_partial_failure_records_evidence_and_restores_hook(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp); env = Environment(fail_step=2); original = env.raw.sim.step
            with self.assertRaisesRegex(RuntimeError, 'INJECTED_STEP_FAILURE'): self.run_block(path, env)
            self.assertEqual(env.raw.sim.steps, 8); self.assertTrue(env.closed)
            self.assertEqual(env.raw.sim.step, original)
            with np.load(path / 'partial_trajectory.npz') as data: self.assertEqual(len(data['fake_value']), 2)
            with np.load(path / 'substeps_partial.npz') as data: self.assertEqual(len(data['physics_step']), 8)
            self.assertFalse((path / 'trajectory.npz').exists())


if __name__ == '__main__': unittest.main(verbosity=2)
