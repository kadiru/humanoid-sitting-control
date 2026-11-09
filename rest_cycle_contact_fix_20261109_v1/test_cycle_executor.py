"""CPU controller-contract tests, not simulation or physical successes."""
import copy
from dataclasses import replace
import math
import unittest
import numpy as np
from cycle_executor import CyclePlan, CycleExecutor, Measurements, pitch_from_root, encode_float32_path


def root(degrees):
    a=math.radians(degrees);c,s=math.cos(a),math.sin(a)
    return [[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]]


def example():
    fingers=np.linspace(-.2,.2,14).tolist()
    nodes=[dict(command=[i*.001]*23+fingers,root=root(pitch),normal=i in (1,2),
                gap=gap,load=load)
           for i,(pitch,gap,load) in enumerate(((0.,.05,0.),(26.,.012,0.),
               (26.,0.,0.),(20.,0.,.10),(3.,0.,.4)))]
    back=copy.deepcopy(nodes[::-1])
    def schedule(rows):
        commands=np.asarray([r['command'] for r in rows[1:]])
        previous=np.asarray([r['command'] for r in rows[:-1]])
        desired=previous[:,:23]+(commands[:,:23]-previous[:,:23])/.1
        return dict(commands=commands.tolist(),desired=desired.tolist())
    return dict(nodes=nodes,return_nodes=back,forward=schedule(nodes),backward=schedule(back))


def compiled(plan=None):
    return CyclePlan(example() if plan is None else plan,[-2.]*23,[2.]*23,
                     hover_node=1,touch_node=2,loaded_node=3,release_node=3)


def measurement(executor, step):
    """Idealized independent input generator with explicit stage values."""
    p=executor.plan;i=executor.index
    if executor.phase=='sit':
        pitch={0:0.,1:26.,2:26.,3:20.,4:3.}[i]
        loaded=i>=3
        lo,hi=(.38,.42) if loaded else (0.,0.)
        separated=i==0
    else:
        pitch={0:3.,1:20.,2:26.,3:26.,4:0.}[i]
        separated=i>=3
        lo,hi=(0.,0.) if separated else (.2,.4)
    a=math.radians(pitch)
    return Measurements(step,(0.,)*23,(math.sin(a),0.,-math.cos(a)),0.,True,0.,True,
                        True,lo,hi,separated)


class Harness:
    def __init__(self, origin=0):
        # Idealized sensor protocol only: independent real-world load and
        # release admissions must both be established before a robot launch.
        self.executor=CycleExecutor(compiled(),current_path_admitted=True,loaded_observer_admitted=True,
                                    release_observer_admitted=True)
        self.previous=self.executor.plan.initial.astype(np.float32)
        self.step=origin;self.rows=[]
    def tick(self, change=None):
        sample=measurement(self.executor,self.step)
        if change:sample=change(sample)
        desired,result=self.executor.command(sample,self.previous)
        applied=self.previous.copy()
        applied[:23]+=np.clip(np.float32(.1)*(np.asarray(desired,np.float32)-applied[:23]),
                              np.float32(-.003),np.float32(.003))
        self.previous=applied;self.step+=4;self.rows.append(result)
        return result
    def until(self,predicate,limit=2001):
        for _ in range(limit):
            row=self.tick()
            if predicate(row):return row
        raise AssertionError('controller did not return within its finite limit')


class ReferenceChecks(unittest.TestCase):
    def test_pitch_uses_frozen_stage_not_standing_or_measured_reanchoring(self):
        sample=Measurements(0,(0.,)*23,(math.sin(math.radians(26)),0.,-math.cos(math.radians(26))),0.,True,0.,True)
        self.assertTrue(sample.packet(pitch_from_root(root(26))).quiet())
        self.assertFalse(sample.packet(pitch_from_root(root(0))).quiet())
        self.assertFalse(sample.packet(pitch_from_root(root(25))).quiet())
    def test_measurement_owns_reused_producer_buffers(self):
        velocity=[0.]*23;gravity=[0.,0.,-1.]
        sample=Measurements(0,velocity,gravity,0.,True,0.,True)
        velocity[-1]=1.;gravity[0]=1.
        self.assertTrue(sample.packet(0.).quiet())
    def test_invalid_imu_cannot_qualify(self):
        sample=Measurements(0,(0.,)*23,(0.,0.,0.),0.,True,0.,True)
        self.assertFalse(sample.available());self.assertFalse(sample.packet(0.).quiet())
    def test_bad_frozen_rotation_is_not_a_reference(self):
        r=root(26);r[0][0]=2.
        with self.assertRaisesRegex(ValueError,'ROTATION'):pitch_from_root(r)
    def test_zero_load_touch_cannot_be_the_loaded_gate(self):
        p=example();p['nodes'][3]['load']=0.
        with self.assertRaisesRegex(ValueError,'POSITIVE_MODEL_SUPPORT'):compiled(p)
    def test_input_plan_mutation_does_not_change_commands(self):
        source=example();p=compiled(source);before=p.forward.copy()
        source['forward']['desired'][0][0]=1.
        np.testing.assert_array_equal(p.forward,before)
        with self.assertRaises(ValueError):p.forward[0,0]=1.
    def test_original_rate_and_inverse_filter_are_checked(self):
        p=example();p['forward']['desired'][0][0]=.5
        with self.assertRaisesRegex(ValueError,'COMMAND_CONTRACT'):compiled(p)
    def test_float32_encoding_prevents_accumulation_in_long_path(self):
        initial = np.full(23, -1.2)
        targets = np.array([initial-(i+1)*.0000244 for i in range(200)])
        encoded = encode_float32_path(targets, initial, np.full(23, -2.), np.full(23, 2.))
        previous = initial.astype(np.float32)
        for desired, target in zip(encoded, targets):
            previous += np.clip(np.float32(.1)*(desired-previous), np.float32(-.003), np.float32(.003))
            np.testing.assert_allclose(previous, target.astype(np.float32), rtol=0, atol=2e-7)
    def test_unrepresentable_path_cannot_relax_rate_limit(self):
        with self.assertRaisesRegex(ValueError,'COMMAND_ALLOWANCE'):
            encode_float32_path([[.004]*23], [0.]*23, np.full(23,-2.), np.full(23,2.))


class ExecutionChecks(unittest.TestCase):
    def test_admissions_are_required(self):
        for path,load in ((False,True),(True,False)):
            with self.assertRaisesRegex(ValueError,'ADMISSION_REQUIRED'):
                CycleExecutor(compiled(),current_path_admitted=path,loaded_observer_admitted=load)
        with self.assertRaisesRegex(ValueError,'INDEPENDENT_RELEASE'):
            CycleExecutor(compiled(),current_path_admitted=True,loaded_observer_admitted=True)
    def test_three_clock_origins_complete_only_sensor_protocol(self):
        for origin in (0,104,5000):
            with self.subTest(origin=origin):
                h=Harness(origin);r=h.until(lambda r:r['sensor_cycle_complete'])
                self.assertFalse(r['aborted']);self.assertFalse(r['physical_success'])
                self.assertEqual(r['quiet_samples'],100)
                self.assertTrue(all(x['phase']=='sit' for x in h.rows[:1000]))
                self.assertGreaterEqual(h.rows[1000]['hold_samples'],250)
                np.testing.assert_array_equal(h.previous[23:],h.executor.plan.initial.astype(np.float32)[23:])
                np.testing.assert_allclose(h.previous,h.executor.plan.initial,rtol=0,atol=2e-7)
    def test_zero_load_contact_pose_allows_bounded_transfer_after_quiet(self):
        h=Harness();h.until(lambda r:r['path_index']==2)
        for _ in range(24):
            self.assertEqual(h.tick()['path_index'],2)
        # This input has zero measured support. It can begin the admitted
        # transfer but must not be described as a qualified supported hold.
        r=h.tick();self.assertEqual(r['path_index'],3);self.assertEqual(r['hold_samples'],0)
        for _ in range(30):
            r=h.tick(lambda s:replace(s,load_lower_bw=0.,load_upper_bw=.02))
            self.assertEqual(r['path_index'],3)
    def test_native_ankle_reading_is_not_replaced_or_relaxed(self):
        h=Harness();h.until(lambda r:r['path_index']==4)
        for _ in range(1000):
            r=h.tick(lambda s:replace(s,raw_joint_velocity=(0.,)*22+(.055,)))
            if r['aborted']:break
        self.assertTrue(r['aborted']);self.assertEqual(r['hold_samples'],0)
        self.assertEqual(r['phase'],'sit')
    def test_hold_must_be_final_and_uninterrupted(self):
        h=Harness()
        for i in range(1001):
            r=h.tick((lambda s:replace(s,geometry_age_s=.22)) if i==999 else None)
        self.assertTrue(r['aborted']);self.assertLess(r['hold_samples'],250)
        self.assertFalse(r['sensor_cycle_complete'])
    def test_release_requires_current_evidence_not_a_timer(self):
        h=Harness();h.until(lambda r:r['phase']=='return' and r['path_index']==3)
        for _ in range(1001):
            r=h.tick(lambda s:replace(s,geometry_separated=False))
            if r['aborted']:break
        self.assertTrue(r['aborted']);self.assertEqual(r['path_index'],3)
    def test_unknown_force_cannot_release(self):
        h=Harness();h.until(lambda r:r['phase']=='return' and r['path_index']==3)
        for _ in range(30):
            r=h.tick(lambda s:replace(s,load_upper_bw=float('nan')))
            self.assertEqual(r['path_index'],3);self.assertEqual(r['quiet_samples'],0)
    def test_skipped_sensor_frame_latches_fault(self):
        h=Harness();h.tick();h.step+=4
        self.assertTrue(h.tick()['aborted']);self.assertTrue(h.tick()['aborted'])
    def test_stale_geometry_pauses_without_skipping_commands(self):
        h=Harness();h.until(lambda r:r['path_index']==1);before=h.previous.copy()
        for _ in range(30):
            r=h.tick(lambda s:replace(s,geometry_age_s=float('nan')))
            self.assertEqual(r['path_index'],1);np.testing.assert_array_equal(h.previous,before)
        for _ in range(24):self.assertEqual(h.tick()['path_index'],1)
        self.assertEqual(h.tick()['path_index'],2)
    def test_double_filter_or_omitted_command_is_rejected(self):
        h=Harness();h.until(lambda r:r['path_index']==1)
        h.previous[:23]=0.
        previous=h.previous.copy();result=h.tick()
        self.assertTrue(result['aborted']);self.assertFalse(result['actuation_allowed'])
        np.testing.assert_array_equal(h.previous,previous)
    def test_fingers_must_be_bitwise_fixed_even_inside_rounding_allowance(self):
        h=Harness();h.previous[23]=np.nextafter(h.previous[23],np.float32(np.inf))
        self.assertTrue(h.tick()['aborted'])


if __name__=='__main__':unittest.main()
