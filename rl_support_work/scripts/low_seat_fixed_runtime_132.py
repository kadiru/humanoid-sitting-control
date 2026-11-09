"""Two measured-geometry fixed-foot approaches with current-sensor guards.

Explicit pre-training controller revision: selected rows replace the old
landing reference after1s with126/128. Unselected rows retain guarded normal3s.
The frozen parent remains additive at15s for selected rows. No physics criteria,
materials, filter, actuator rate, finger commands or optimizer are changed.
"""
from pathlib import Path
import inspect,textwrap,json,shutil
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel
from low_seat_fixed_foot_plan_126 import build_plan,measured,route
from low_seat_gated_command_128 import GatedCommandPath,support_limit
from low_seat_measured_guard_130 import MeasuredGuard
from low_seat_dual_view_40 import fallback,DOWN
from low_seat_normal_timing_controller import install as normal_install

def install(harness,landing_config,config):
    normal_install(harness,landing_config)
    # Reuse exactly the independently checked CURRENT-view selection40, without
    # installing its rejected staged motion or its different command-rate cap.
    import run_supported_transition as transition
    original=transition.TransitionEnv._observe;source=textwrap.dedent(inspect.getsource(original))
    needle="for i in range(self.num_envs):self.geometry[i]=self.trackers[i].update(q[i],gravity[i],gyro[i],measurement=measurements[i])"
    assert source.count(needle)==1
    insertion="measurements=fallback(self,measurements,q,gravity,force)\n    "
    replacement=source.replace(needle,insertion+needle)
    assert replacement.replace(insertion,'')==source
    scope=dict(original.__globals__,fallback=fallback);exec(replacement,scope)
    transition.TransitionEnv._observe=scope['_observe']
    build=harness.parent_harness.build
    def with_camera(*args,**kw):
        cfg,goals,resets=build(*args,**kw)
        cfg.scene.diagnostic_down_camera=harness.base.CameraCfg(prim_path='{ENV_REGEX_NS}/Robot/pelvis/DiagnosticDownCamera',width=128,height=96,data_types=['distance_to_image_plane'],update_period=1.e6,
            offset=harness.base.CameraCfg.OffsetCfg(pos=tuple(DOWN[:3,3]),rot=tuple(Rotation.from_matrix(DOWN[:3,:3]).as_quat()),convention='ros'),
            spawn=harness.base.sim.PinholeCameraCfg(focal_length=18.,horizontal_aperture=24.,clipping_range=(.05,4.)))
        return cfg,goals,resets
    harness.parent_harness.build=with_camera
    Parent=harness.FeedbackEnv
    class FixedEnv(Parent):
        def __init__(self,*args,**kwargs):
            self.fixed_ready=False;self.staged_ready=False
            self.fixed_states={};self.fixed_plans=[];self.fixed_records=[];self.fixed_guard_records=[]
            self.dual_records=[];self.dual_details=[]
            super().__init__(*args,**kwargs)
            self.cooked_model=BodyModel(Path(config['cooked_model']))
            self.cooked_meta=json.loads(Path(config['cooked_model']).read_text())
            self.fixed_inertia=json.loads(Path(config['inertia']).read_text())
            assert list(self.cooked_model.names)==list(self.model.names)==list(self.robot.joint_names)
            self.repair_duration=torch.full((self.num_envs,),5.,device=self.device)
            self.fixed_ready=True;self.staged_ready=True;self.fixed_observed_tick=self.t
        def _reset_rows(self,ids):
            if self.fixed_ready:
                for i in ids.cpu().tolist():self.fixed_states.pop(i,None)
                self.repair_duration[ids]=5.
            return super()._reset_rows(ids)
        def _observe(self,force=False):
            super()._observe(force)
            if not self.fixed_ready or not self.planning_enabled:return
            self.fixed_observed_tick=self.t
            pending=(self.near_start<0)&(self.warming==0)&(self.episode_length_buf==50)
            q=self.robot.data.joint_pos.torch.cpu().numpy();g=self.robot.data.projected_gravity_b.torch.cpu().numpy()
            for i in pending.nonzero().flatten().tolist():
                geometry=self.geometry[i].copy()
                if not geometry[11] or geometry[12]>.5:continue
                reference=np.array([float(self.estimated_root_x[i]),0.,0.])
                plane=measured(self.cooked_model,q[i],g[i],geometry,reference,self.calibration)
                choice=route(plane[3],plane[4],config)
                if choice is None:continue
                command=self.previous[i].cpu().numpy().copy()
                inputs=dict(q=q[i].tolist(),gravity=g[i].tolist(),geometry=geometry.tolist(),reference=reference.tolist(),command=command.tolist(),endpoint=choice['endpoint'])
                plan=build_plan(self.cooked_model,self.cooked_meta,self.fixed_inertia,self.calibration,q[i],g[i],geometry,reference,command,choice['endpoint'])
                self.fixed_plans.append(dict(case=i,tick=int(self.episode_length_buf[i]),route=choice['name'],inputs=inputs,plan=plan))
                if not plan['model_passed']:continue
                engine=GatedCommandPath(plan,self.cooked_model);guard=MeasuredGuard(self.cooked_model,plan,geometry)
                self.fixed_states[i]=dict(engine=engine,guard=guard,start=self.t,tick=self.t,target=command.copy(),plan=plan)
                # Prevent a second old landing plan at10s. These constant slots
                # are compatibility bookkeeping; reference() below supplies the
                # actual current guarded target to both motors and414 inputs.
                self.near_paths[i]=torch.tensor(np.tile(command,(25,1)),device=self.device)
                self.near_q[i]=torch.tensor(np.tile(q[i],(25,1)),device=self.device)
                self.near_start[i]=self.episode_length_buf[i];self.repair_duration[i]=14.
                initial=plan['path'][0]['root']
                self.plan_audit.append(dict(case=i,attempt=int(self.attempt[i]),tick=int(self.episode_length_buf[i]),stage=int(self.blend[i]),
                    root_path=[initial]*25,positive_candidate_applied=True,landing_repair_applied=False,
                    fixed_sensor_path_132=True,actual_plan_file='fixed_plans_132.json',compatibility_slots_are_not_motion=True))
            self._augment()
        def reference(self):
            ref=super().reference()
            if not self.fixed_ready or not self.planning_enabled:return ref
            data=self.robot.data
            q=data.joint_pos.torch.cpu().numpy();g=data.projected_gravity_b.torch.cpu().numpy()
            qd=data.joint_vel.torch.cpu().numpy();gyro=data.root_ang_vel_b.torch.cpu().numpy()
            for i,state in self.fixed_states.items():
                if self.warming[i]:continue
                if state['tick']!=self.t and self.fixed_observed_tick==self.t:
                    state['tick']=self.t;engine=state['engine'];a=state['guard'].observe(q[i],g[i],self.geometry[i])
                    conditions=self.shear_records[-1]['condition'][i] if self.shear_records else [np.inf,np.inf]
                    support=dict(max_normal_bw=0.,plane_consistent=False)
                    progress=False;tracking=None
                    if a['available']:
                        support=support_limit(bool(self.geometry[i,11]),float(self.geometry[i,12]),a['plane_height_error_m'],a['plane_slope_error_deg'],
                            conditions,self.filtered_wrench[i],self.wrench_observer.weight,a['normal_z'],
                            float(np.max(abs(qd[i,engine.ids]))),float(np.linalg.norm(gyro[i])),a['pelvis_gap_m'],a['patch_margin_m'],a['housing_clearance_m'])
                        Q=np.array([e['q'] for e in state['plan']['path']])
                        expected=np.array([np.interp(engine.position,engine.arc,Q[:,j]) for j in engine.ids])
                        tracking=float(np.max(abs(q[i,engine.ids]-expected)))
                        progress=bool(support['plane_consistent'] and a['anchor_valid'] and a['housing_clearance_m']>=.003
                            and (a['pelvis_gap_m']>=.02 or a['patch_margin_m']>=.015) and tracking<=.12)
                    state['target']=engine.advance(progress,support['max_normal_bw'])
                    self.fixed_guard_records.append(dict(tick=self.t,case=i,geometry=self.geometry[i].tolist(),
                        measured_guard=a,estimated_support=support,tracking_error_rad=tracking,progress_allowed=progress,
                        command_path=dict(engine.last_info)))
                ref[i]=torch.tensor(state['target'],device=self.device,dtype=ref.dtype)
            return ref
        def _physics_step(self,actions):
            if not self.fixed_ready:return super()._physics_step(actions)
            before=self.previous.clone();selected=torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
            selected[list(self.fixed_states)]=True
            cap=self.profile_values['command_step'][self.index]
            assert torch.all(abs(cap[selected]-.003)<2e-10)
            result=super()._physics_step(actions)
            self.fixed_records.append(dict(t=float(self.t*.02),selected=selected.cpu().numpy(),
                before=before.cpu().numpy(),after=self.previous.cpu().numpy().copy(),cap=cap.cpu().numpy().copy(),actions=actions.cpu().numpy().copy()))
            return result
        def close(self):
            super().close()
            (self.out/'fixed_plans_132.json').write_text(json.dumps(self.fixed_plans))
            (self.out/'fixed_guard_records_132.json').write_text(json.dumps(self.fixed_guard_records))
            for name,records in [('fixed_ticks_132',self.fixed_records),('dual_view_inputs_132',self.dual_records)]:
                if records:np.savez_compressed(self.out/(name+'.npz'),**{k:np.stack([r[k] for r in records]) for k in records[0]})
            (self.out/'dual_view_details_132.json').write_text(json.dumps(self.dual_details))
            (self.out/'fixed_contract_132.json').write_text(json.dumps(dict(scope=__doc__,config=config,
                selected_control='Current measured126 path;128 nominal inverse-filter reference, current130 geometry/fixed-foot guard; estimated load gating with .02BW reserve. Old selected landing/balance corrections are replaced, not silently composed. Unselected guardednormal3s unchanged in source.',
                parent='Unchanged frozen59117 additive at15s selected, original11/15 routing elsewhere; no early61111 pulse in this isolating probe.',
                sensors='Perception still uses original raw visual/self-mask model; cooked50 only in planning/clearance. Downfallback40 only current selected missing-rear frames.',
                rate_rad_s=.15,filter_alpha=.1,desired_margin_rad=.03,locked_prefix_s=1.,optimizer_updates=0,
                unselected_exactness='Measure; not assumed. Prior rendered-scene divergence remains possible.',
                observer_original=source,observer_replacement=replacement),indent=2))
            for n in ['low_seat_fixed_runtime_132.py','low_seat_fixed_foot_plan_126.py','low_seat_gated_command_128.py','low_seat_measured_guard_130.py']:
                shutil.copy2(Path(__file__).with_name(n),self.out/'source'/n)
    harness.FeedbackEnv=FixedEnv
