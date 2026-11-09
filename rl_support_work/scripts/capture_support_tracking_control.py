"""Frozen actor evaluation with correctly timed tracked, observed seat patches."""
import capture_support_control_timing as timing
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from humanoid_sitting.support_landing_geometry import LandingGeometryTracker
from humanoid_sitting.support_tracking_features import measured_tracking_features
experiment=timing.experiment
capture=timing.capture
assert timing.options.sync_scope=='both','Tracked control requires current, matched camera/proprioception'


class TrackingControlEnv(timing.TimingAuditEnv):
    def __init__(self,*args,**kwargs):
        self.trackers={};self.shadow_records=[];self.shadow_times=[];self.tracking_tick=-1
        self.feature_audit=[]
        super().__init__(*args,**kwargs)

    def _admit(self,ids):
        super()._admit(ids)
        for i in ids.tolist():
            if not self.ready[i]:continue
            observer=self.observers[i].observer
            tracker=LandingGeometryTracker(experiment.BODY,observer.K,observer.T_P_C,observer.seat_height)
            f=next(x for x in reversed(self.startup_frames) if x['case']==i)
            measured=experiment.estimate_body_frame(f['depth'],f['K'],self.mount,self.quat,f['gravity'])
            tracker.initialize_from_measured_floor((self.t+1)*.02,f['q'],f['gravity'],measured['floor_height_body_m'])
            self.trackers[i]=tracker

    def _observe(self,update_depth):
        super()._observe(update_depth)
        if self.t<=0 or self.t==self.tracking_tick:return
        self.tracking_tick=self.t;f=self.sensor_frames[-1]
        assert abs(f['t']-self.t*.02)<1e-7
        images={x['case']:x['depth'] for x in self.camera_frames[-self.num_envs:] if abs(x['t']-f['t'])<1e-7}
        for i in range(self.num_envs):
            if i not in self.trackers:
                r=dict(stamp=f['t'],valid=False,clearance_m=None,source='unavailable',reason='startup_refused')
            else:
                start=time.perf_counter();tracker=self.trackers[i]
                r=tracker.update(f['t'],f['q'][i],f['gravity'][i],f['gyro'][i],images.get(i))
                if i in images:
                    values,detail=measured_tracking_features(tracker,r,images[i],f['q'][i],f['gravity'][i])
                    legacy=self.live[i].cpu().numpy().copy()
                    self.live[i]=torch.as_tensor(values,device=self.device)
                    self.feature_audit.append(dict(t=f['t'],case=i,source=r['source'],legacy=legacy.tolist(),fed=values.tolist(),**detail))
                self.shadow_times.append(time.perf_counter()-start)
            self.shadow_records.append(dict(case=i,**r))
        # Only the residual actor consumes these ten channels. Its frozen nominal
        # controller has already advanced exactly once from the original packet.
        self.obs['policy'][:,132:142]=self.live
        # Evaluation-only archive; none of these values enter the tracker or actor.
        labels=experiment.physical_metrics(self.raw)
        for name in ('peak_force','vz','leg_rms_nm','tilt2'):
            self.label_frames[-1][name]=labels[name].cpu().numpy().copy()

    def close(self):
        for filename,rows in [('shadow_records.jsonl',self.shadow_records),('tracking_feature_audit.jsonl',self.feature_audit)]:
            with (self.out/filename).open('w') as f:
                for row in rows:f.write(json.dumps(row)+'\n')
        (self.out/'shadow_runtime.json').write_text(json.dumps(dict(samples=len(self.shadow_records),
            median_ms=float(np.median(self.shadow_times)*1000),p95_ms=float(np.quantile(self.shadow_times,.95)*1000),
            total_processing_s=float(sum(self.shadow_times)),fed_to_controller=True,
            note='Tracker runs at 50 Hz; frozen actor keeps 10 Hz complete measured-patch contract and explicit sample age.'),indent=2)+'\n')
        super().close()
        p=self.out/'calibration.json';metadata=json.loads(p.read_text())
        metadata['original_controller_inputs_unchanged']=False
        metadata['tracking_controller_inputs']='Ten-channel observed patch from new tracker at 10 Hz, with sample age; same weights'
        p.write_text(json.dumps(metadata,indent=2)+'\n')


original_cfg=experiment.DeploymentEnvCfg
def tracking_cfg(**kwargs):
    cfg=original_cfg(**kwargs);out=Path(capture.independent.args.output);m=json.loads((out/'manifest.json').read_text())
    for source in [Path(__file__).resolve(),experiment.ROOT/'sitting_task/src/humanoid_sitting/support_tracking_features.py',
                   experiment.ROOT/'sitting_task/src/humanoid_sitting/support_landing_geometry.py',
                   experiment.ROOT/'scripts/test_support_tracking_features.py']:
        shutil.copy2(source,out/'source'/source.name)
        m['hashes'][str(source.relative_to(experiment.ROOT))]=hashlib.sha256(source.read_bytes()).hexdigest()
    m['tracking_control']=dict(training=False,policy_receives_tracker=True,
        tracker_hz=50,actor_measured_patch_hz=10,unobserved_bounds='withhold complete ten-channel feature',
        unchanged='all weights, 109-channel sensor packet, nominal scaffold, motor bounds and gains')
    m['timing_audit']['tracker_fed_to_actor']=True
    (out/'manifest.json').write_text(json.dumps(m,indent=2)+'\n');return cfg
experiment.SupportVecEnv=TrackingControlEnv
experiment.DeploymentEnvCfg=tracking_cfg
if __name__=='__main__':experiment.main()
