"""Project-owned render-only synchronization audit; no physics steps added."""
import argparse
import sys
p=argparse.ArgumentParser(add_help=False);p.add_argument('--render-flushes',type=int,default=1);p.add_argument('--capture-refresh',choices=['render','pump','due'],default='due')
options,rest=p.parse_known_args();sys.argv=[sys.argv[0],*rest]
import capture_support_landing as capture
import json
import shutil
from pathlib import Path
import torch
import hashlib
from humanoid_sitting.support_camera import acquire_current_camera

class SynchronizedCaptureEnv(capture.CaptureEnv):
    def __init__(self,*args,**kwargs):
        self.render_audit=[]
        super().__init__(*args,**kwargs)

    def _fresh_camera(self,name):
        acquire_current_camera(self.raw.scene[name])

    def _admit(self,ids):
        if options.capture_refresh=='due':
            self._fresh_camera('startup_camera')
            self._fresh_camera('live_camera')
        super()._admit(ids)

    def _observe(self,update_depth):
        if self.t>0 and (update_depth or self.pending.any()) and (~self.warming).any():
            sim=self.raw.sim
            step=sim.get_physics_step_count();q=self.robot.data.joint_pos.torch.clone()
            sim.render_context.update_scene_state(step)
            if options.capture_refresh=='due':
                self._fresh_camera('live_camera')
            elif options.capture_refresh=='render':
                for _ in range(options.render_flushes):sim.render()
            else:
                import omni.kit.app
                previous=sim.get_setting('/app/player/playSimulations')
                try:
                    sim.set_setting('/app/player/playSimulations',False)
                    for _ in range(options.render_flushes):omni.kit.app.get_app().update()
                finally:sim.set_setting('/app/player/playSimulations',previous)
            assert step==sim.get_physics_step_count(), 'Render advanced physics'
            assert torch.equal(q,self.robot.data.joint_pos.torch), 'Render changed joints'
            self.render_audit.append(dict(t=self.t*.02,physics_step=step,render_generation=sim.render_generation,
                camera_timestamp=self.raw.scene['live_camera']._timestamp.numpy().tolist(),
                camera_last_update=self.raw.scene['live_camera']._timestamp_last_update.numpy().tolist(),
                camera_frame=self.raw.scene['live_camera'].frame.torch.cpu().tolist()))
        super()._observe(update_depth)
        if self.render_audit and options.capture_refresh=='due':
            last=self.render_audit[-1]
            assert all(abs(x-y)<1e-6 for x,y in zip(last['camera_timestamp'],last['camera_last_update'])), 'Stale capture timestamp'

    def close(self):
        (self.out/'render_audit.json').write_text(json.dumps(dict(flushes=options.render_flushes,refresh=options.capture_refresh,checks=self.render_audit),indent=2)+'\n')
        manifest=json.loads((self.out/'manifest.json').read_text())
        for source in [Path(__file__).resolve(),capture.experiment.ROOT/'sitting_task/src/humanoid_sitting/support_camera.py']:
            shutil.copy2(source,self.out/'source'/source.name)
            manifest['hashes'][str(source.relative_to(capture.experiment.ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
        manifest['capture_refresh']=options.capture_refresh
        (self.out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        super().close()

capture.experiment.SupportVecEnv=SynchronizedCaptureEnv
if __name__=='__main__':capture.experiment.main()
