"""Evaluation-only camera timing factorial with complete controller telemetry."""
import argparse
import sys
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--sync-scope',choices=['none','startup','live','both'],required=True)
options,rest=p.parse_known_args();sys.argv=[sys.argv[0],*rest]
import capture_support_landing_sync as sync
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
capture=sync.capture
experiment=capture.experiment


class TimingAuditEnv(sync.SynchronizedCaptureEnv):
    def __init__(self,*args,**kwargs):
        self.control_records=[];self.admission_geometry=[]
        super().__init__(*args,**kwargs)

    def _admit(self,ids):
        if options.sync_scope in ('startup','both'):self._fresh_camera('startup_camera')
        if options.sync_scope in ('live','both'):self._fresh_camera('live_camera')
        capture.CaptureEnv._admit(self,ids)
        self.admission_geometry.append(dict(t=(self.t+1)*.02,ids=ids.tolist(),
            geometry=self.sensor.geometry[ids].cpu().tolist(),ready=self.ready[ids].cpu().tolist()))

    def _observe(self,update_depth):
        if options.sync_scope in ('live','both'):
            sync.SynchronizedCaptureEnv._observe(self,update_depth)
        else:capture.CaptureEnv._observe(self,update_depth)

    def step(self,actions):
        record=dict(t=self.t*.02,policy=self.obs['policy'].cpu().numpy().copy(),
            actions=actions.cpu().numpy().copy(),previous=self.previous.cpu().numpy().copy(),
            ready=self.ready.cpu().numpy().copy())
        result=super().step(actions)
        # The last step resets the environment. The actual pre-reset command is
        # reconstructed using exactly the original bounds from saved inputs.
        record['command_target']=np.clip(record['policy'][:,109:132]+.08*np.clip(record['actions'],-1,1)*record['ready'][:,None],
            record['previous'][:,self.ids]-.05,record['previous'][:,self.ids]+.05)
        limits=self.robot.data.joint_pos_limits.torch[:,self.ids].cpu().numpy()
        record['command_target']=np.clip(record['command_target'],limits[...,0],limits[...,1])
        self.control_records.append(record)
        return result

    def close(self):
        np.savez_compressed(self.out/'controller.npz',**{k:np.stack([x[k] for x in self.control_records]) for k in self.control_records[0]})
        (self.out/'admission_geometry.json').write_text(json.dumps(self.admission_geometry,indent=2)+'\n')
        super().close()


original_cfg=experiment.DeploymentEnvCfg
def audit_cfg(**kwargs):
    cfg=original_cfg(**kwargs);out=Path(capture.independent.args.output)
    m=json.loads((out/'manifest.json').read_text());source=Path(__file__).resolve()
    shutil.copy2(source,out/'source'/source.name)
    m['hashes'][str(source.relative_to(experiment.ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    m['timing_audit']=dict(sync_scope=options.sync_scope,training=False,tracker_fed_to_actor=False,
        full_controller_telemetry='controller.npz at 50 Hz, before each command',
        target_record='reconstructed from logged nominal, action, previous and actual joint bounds')
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');return cfg
experiment.SupportVecEnv=TimingAuditEnv
experiment.DeploymentEnvCfg=audit_cfg
if __name__=='__main__':experiment.main()
