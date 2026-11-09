"""Behavioral checks: stale, unknown, unquiet and out-of-order data fail closed."""
import unittest
from dataclasses import replace
from contact_gate import Packet,ContactGate

class Checks(unittest.TestCase):
    def setUp(self):
        self.g=ContactGate(100,200);self.step=0
        self.p=Packet(0,(0.,)*23,0.,0.,True,0.,True)
    def update(self,index,**changes):
        self.step+=4
        return self.g.update(index,replace(self.p,physics_step=self.step,**changes))
    def hover(self):
        for _ in range(24):self.assertFalse(self.update(100)['advance'])
        self.assertTrue(self.update(100)['advance'])
    def test_requires_fresh_quiet_at_hover(self):
        for _ in range(30):self.update(0)
        self.hover()
    def test_motion_does_not_require_quiet_every_control(self):
        self.hover()
        self.assertTrue(self.update(101,raw_joint_velocity=(.10,)*23)['advance'])
    def test_no_contact_from_timer_or_unknown_force(self):
        self.hover()
        for _ in range(40):self.assertFalse(self.update(200)['advance'])
        self.assertFalse(self.g.contact_confirmed)
    def test_raw_ankle_failure_cannot_be_smoothed_away(self):
        self.hover()
        for _ in range(30):
            self.assertFalse(self.update(200,raw_joint_velocity=(0.,)*22+(.055,),
                load_interval_admitted=True,load_lower_bw=.52,load_upper_bw=.56)['advance'])
        self.assertEqual(self.g.contact_quiet,0)
    def test_measured_load_and_quiet_need_25_consecutive_samples(self):
        self.hover()
        loaded=dict(load_interval_admitted=True,load_lower_bw=.05,load_upper_bw=.09)
        for _ in range(24):self.assertFalse(self.update(200,**loaded)['advance'])
        self.update(200,**loaded,geometry_age_s=.22)
        for _ in range(24):self.assertFalse(self.update(200,**loaded)['advance'])
        self.assertTrue(self.update(200,**loaded)['advance'])
    def test_stale_packet_aborts(self):
        self.update(0)
        self.assertTrue(self.g.update(0,replace(self.p,physics_step=self.step))['aborted'])
    def test_cannot_skip_hover(self):
        self.assertTrue(self.update(101)['aborted'])
    def test_cannot_skip_contact(self):
        self.hover();self.assertTrue(self.update(201)['aborted'])
    def test_nonfinite_sensor_does_not_qualify(self):
        self.assertFalse(self.update(100,raw_joint_velocity=(float('nan'),)*23)['advance'])
    def test_equality_threshold_is_not_relaxed(self):
        p=replace(self.p,raw_joint_velocity=(.05,)*23)
        self.assertTrue(p.quiet());self.assertFalse(replace(p,raw_joint_velocity=(.05000001,)*23).quiet())
class ExecutorChecks(unittest.TestCase):
    def test_unadmitted_loaded_observer_blocks_runtime(self):
        from contact_executor import ContactExecutor
        with self.assertRaisesRegex(ValueError,'ADMISSION_REQUIRED'):
            ContactExecutor({},hover_index=1,contact_index=2,current_path_admitted=True,loaded_observer_admitted=False)
    def test_pause_does_not_skip_or_modify_waypoints(self):
        from contact_executor import ContactExecutor
        path={'forward':{'desired':[[.001]*23,[.002]*23,[.003]*23]}}
        e=ContactExecutor(path,hover_index=1,contact_index=2,current_path_admitted=True,loaded_observer_admitted=True)
        p=Packet(0,(0.,)*23,0.,0.,True,0.,True)
        q,d=e.command(p,[0.]*37);self.assertEqual(q,[.001]*23)
        for tick in range(1,25):
            q,d=e.command(replace(p,physics_step=4*tick),[.001]*37)
            self.assertEqual(e.index,1);self.assertEqual(q,[.001]*23)
        q,d=e.command(replace(p,physics_step=100),[.001]*37)
        self.assertEqual(e.index,2);self.assertEqual(q,[.002]*23)
        q,d=e.command(replace(p,physics_step=104),[.002]*37)
        self.assertEqual(e.index,2);self.assertEqual(q,[.002]*23)
if __name__=='__main__':unittest.main()
