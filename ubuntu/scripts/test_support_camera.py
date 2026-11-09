"""Regression test for stale period-gated frames and cleanup on failure."""
import unittest
from types import SimpleNamespace
from humanoid_sitting.support_camera import acquire_current_camera
class CameraTests(unittest.TestCase):
 def test_request_replaces_old_image_without_advancing_clock(self):
    c=SimpleNamespace(cfg=SimpleNamespace(update_period=.1),t=13.,last=12.905,image='old')
    def update(dt,force_recompute):
        c.t+=dt
        if force_recompute and c.t-c.last+1e-6>=c.cfg.update_period:c.image='current';c.last=c.t
    c.update=update;c.update(0.,force_recompute=True)
    self.assertEqual(c.image,'old')
    acquire_current_camera(c)
    self.assertEqual((c.image,c.t,c.last,c.cfg.update_period),('current',13.,13.,.1))
 def test_capture_failure_restores_schedule(self):
    def fail(*args,**kwargs):raise RuntimeError('camera disconnected')
    c=SimpleNamespace(cfg=SimpleNamespace(update_period=.1),update=fail)
    with self.assertRaises(RuntimeError):acquire_current_camera(c)
    self.assertEqual(c.cfg.update_period,.1)
if __name__=='__main__':unittest.main()
