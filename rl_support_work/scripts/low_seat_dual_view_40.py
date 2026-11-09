"""Pre-training measured-plane fallback for the unchanged staged36 motion.
Prefer the original current rear fit. Only selected staged rows with no rear
fit may use an independently fitted CURRENT downward exposure, with unchanged
association/conditioning. No truth or replayed geometry is used for selection.
"""
from pathlib import Path
import inspect,textwrap,json,shutil
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.supported_surface import observed_support
from low_seat_staged_controller_36 import install as staged_install
DOWN=np.eye(4);DOWN[:3,:3]=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]]);DOWN[:3,3]=[-.22,0.,.55]

def fallback(self,measurements,q,gravity,force=False):
    if not (force or self.t%10==0):return measurements
    if not getattr(self,'staged_ready',False) or not hasattr(self,'repair_duration'):return measurements
    selected=(self.repair_duration>5.).cpu().numpy()&(self.warming.cpu().numpy()==0)
    if not selected.any():return measurements
    rear=list(measurements);needed=selected&np.array([x is None for x in rear])
    # Priors are measured/propagated tracker values, never labels or saved fits.
    prior=np.full((self.num_envs,13),np.nan,np.float32)
    for i,t in enumerate(self.trackers):
        if t.value is not None:prior[i]=t.value.copy()
    from run_supported_transition import base
    rear_cam=self.raw.scene['rear_camera'];depth=rear_cam.data.output['distance_to_image_plane'].torch.cpu().numpy().copy();K=rear_cam.data.intrinsic_matrices.torch.cpu().numpy().copy()
    down_depth=np.full_like(depth,np.nan);down_K=np.full_like(K,np.nan)
    source=np.array([0 if x is not None else -1 for x in rear],np.int8);detail=[{} for _ in range(self.num_envs)]
    clock=self.raw.sim.get_physics_step_count()
    if needed.any():
        cam=self.raw.scene['diagnostic_down_camera'];base.acquire_current_camera(cam)
        assert self.raw.sim.get_physics_step_count()==clock
        down_depth=cam.data.output['distance_to_image_plane'].torch.cpu().numpy().copy();down_K=cam.data.intrinsic_matrices.torch.cpu().numpy().copy()
        for i in np.flatnonzero(needed):
            fit,detail[i]=observed_support(self.model,down_K[i],DOWN,down_depth[i],q[i],gravity[i],prior=self.trackers[i].value)
            if fit is not None:measurements[i]=fit;source[i]=1
    assert all(measurements[i] is rear[i] for i in range(self.num_envs) if not needed[i])
    encode=lambda rows:np.array([np.full(13,np.nan,np.float32) if x is None else x.copy() for x in rows])
    self.dual_records.append(dict(t=float(self.t*.02),selected=selected.copy(),needed=needed.copy(),q=q.copy(),gravity=gravity.copy(),prior=prior,rear_depth=depth,rear_K=K,down_depth=down_depth,down_K=down_K,rear_fit=encode(rear),chosen_fit=encode(measurements),source=source,physics_clock=clock))
    self.dual_details.append(dict(time=float(self.t*.02),details=detail))
    return measurements

def install(harness,config):
    staged_install(harness,config)
    import run_supported_transition as transition
    original=transition.TransitionEnv._observe;src=textwrap.dedent(inspect.getsource(original))
    needle='for i in range(self.num_envs):self.geometry[i]=self.trackers[i].update(q[i],gravity[i],gyro[i],measurement=measurements[i])'
    assert src.count(needle)==1
    insertion='measurements=_dual_view_fallback_40(self,measurements,q,gravity,force)\n    '
    replacement=src.replace(needle,insertion+needle)
    assert replacement.replace(insertion,'')==src
    scope=dict(original.__globals__);scope['_dual_view_fallback_40']=fallback;exec(replacement,scope);transition.TransitionEnv._observe=scope['_observe']
    build=harness.parent_harness.build
    def with_down_camera(*args,**kw):
        cfg,goals,resets=build(*args,**kw);assert cfg.scene.rest_camera is None
        cfg.scene.diagnostic_down_camera=harness.base.CameraCfg(prim_path='{ENV_REGEX_NS}/Robot/pelvis/DiagnosticDownCamera',width=128,height=96,data_types=['distance_to_image_plane'],update_period=1.e6,
            offset=harness.base.CameraCfg.OffsetCfg(pos=tuple(DOWN[:3,3]),rot=tuple(Rotation.from_matrix(DOWN[:3,:3]).as_quat()),convention='ros'),
            spawn=harness.base.sim.PinholeCameraCfg(focal_length=18.,horizontal_aperture=24.,clipping_range=(.05,4.)))
        return cfg,goals,resets
    harness.parent_harness.build=with_down_camera
    Parent=harness.FeedbackEnv
    class DualViewEnv(Parent):
        def __init__(self,*a,**kw):
            self.dual_records=[];self.dual_details=[]
            super().__init__(*a,**kw)
        def close(self):
            super().close()
            rr=self.dual_records
            if rr:np.savez_compressed(self.out/'dual_view_inputs_40.npz',**{k:np.stack([r[k] for r in rr]) for k in rr[0]})
            (self.out/'dual_view_details_40.json').write_text(json.dumps(self.dual_details))
            (self.out/'dual_view_contract_40.json').write_text(json.dumps(dict(scope='PRE-TRAINING measured-camera fallback. Same staged36 motion, bounds, pause guard, parent timing and physical criteria. Prefer rear fit; only measured-selected staged rows with no rear fit may use current down exposure with unchanged prior association/conditioning. No substituted baseline sensors or truth. Adding sensor coverage is not physical success.',downward_mount=DOWN.tolist(),sensor_channels=414,feedback_channels=417,optimizer_updates=0,old_observe=src,new_observe=replacement,source_transform_exact=True,unselected_measurement_source='Original rear only; full-trajectory equality is measured, not assumed.'),indent=2))
            shutil.copy2(__file__,self.out/'source'/Path(__file__).name)
    harness.FeedbackEnv=DualViewEnv
