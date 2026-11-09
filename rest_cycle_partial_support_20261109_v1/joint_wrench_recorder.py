"""Shadow ankle F/T sensor experiment. No readings reach the actor.

The recorded joint wrench is total transmitted force, NOT isolated motor torque.
This channel would require suitable hardware instrumentation on a real robot.
"""
import numpy as np

class JointWrenchRecorder:
    def __init__(self,env,support):
        self.env=env;self.support=support;self.rows=[];self.original=None
    def install(self):
        self.original=self.env.raw.sim.step
        def step(*a,**kw):
            out=self.original(*a,**kw);self.record();return out
        self.env.raw.sim.step=step
    def uninstall(self):
        if self.original is not None:self.env.raw.sim.step=self.original;self.original=None
    def record(self):
        import warp as wp
        def array(x):return wp.to_torch(x).detach().cpu().numpy().copy()
        env=self.env;v=env.robot.root_view
        force=[]
        for sensor,_,_ in self.support.CONTACTS:
            cv=env.raw.scene[sensor].contact_view
            normal=array(cv.get_contact_force_matrix(dt=.005)).reshape(-1,3).sum(0)
            f,_,counts,starts=cv.get_friction_data(dt=.005)
            f=array(f).reshape(-1,3);counts=array(counts).reshape(-1);starts=array(starts).reshape(-1)
            assert len(counts)==1
            count,start=int(counts[0]),int(starts[0])
            assert 0<=count<32 and (count==0 or 0<=start<=len(f)-count)
            force.append(normal+(f[start:start+count].sum(0) if count else 0.))
        self.rows.append(dict(physics_step=np.array(env.raw.sim.get_physics_step_count()),
            joint_wrench=array(v.get_link_incoming_joint_force()),
            link_transform=array(v.get_link_transforms()),velocity=array(v.get_link_velocities()),
            mass=array(v.get_masses()),q=array(v.get_dof_positions()),qd=array(v.get_dof_velocities()),
            truth_contact_force=np.asarray(force)))
    def save(self,path):
        assert 0<len(self.rows)<=200
        np.savez_compressed(path,**{k:np.stack([r[k] for r in self.rows]) for k in self.rows[0]})
