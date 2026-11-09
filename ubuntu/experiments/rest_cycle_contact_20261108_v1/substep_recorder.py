"""Post-physics readout only; original stepping and sensors are unchanged.

Captures every physics step, not a final-substep value multiplied by20ms.
Uses the engine's own joint ordering for effort and current COM velocities for
momentum. Inputs go only to offline audits, never to the motion controller.
"""
import numpy as np

class SubstepRecorder:
    def __init__(self,env,support):
        self.env=env;self.support=support;self.rows=[];self.original=None

    def install(self):
        assert self.original is None
        self.original=self.env.raw.sim.step
        def step(*args,**kwargs):
            result=self.original(*args,**kwargs)
            self.record()
            return result
        self.env.raw.sim.step=step

    def uninstall(self):
        if self.original is not None:
            self.env.raw.sim.step=self.original;self.original=None

    def record(self):
        import warp as wp
        def array(a):return wp.to_torch(a).detach().cpu().numpy().copy()
        robot=self.env.robot;v=robot.root_view;raw=self.env.raw
        # Read tensors directly: IsaacLab cached state has not been scene-updated
        # yet at this post-step hook. No additional simulation/update is called.
        velocity=array(v.get_link_velocities())
        mass=array(v.get_masses())
        normals=[];frictions=[];counts=[]
        for sensor,_,_ in self.support.CONTACTS:
            cv=raw.scene[sensor].contact_view
            normals.append(array(cv.get_contact_force_matrix(dt=.005)).reshape(-1,3).sum(0))
            force,_,num,start=cv.get_friction_data(dt=.005)
            force=array(force).reshape(-1,3);num=array(num).reshape(-1);start=array(start).reshape(-1)
            assert len(num)==1
            count,offset=int(num[0]),int(start[0]);assert 0<=count<32 and (count==0 or 0<=offset<=len(force)-count)
            frictions.append(np.zeros(3) if count==0 else force[offset:offset+count].sum(0));counts.append(count)
        self.rows.append(dict(physics_step=np.array(raw.sim.get_physics_step_count()),
            velocity=velocity,mass=mass,normal=np.asarray(normals),friction=np.asarray(frictions),
            all_body_normal=array(raw.scene['all_contacts'].contact_view.get_net_contact_forces(dt=.005)),
            friction_counts=np.asarray(counts),projected_effort=array(v.get_dof_projected_joint_forces()),
            drive_request=array(v.get_dof_actuation_forces())))

    def save(self,path):
        assert len(self.rows)<=4000
        np.savez_compressed(path,**{k:np.stack([r[k] for r in self.rows]) for k in self.rows[0]})
