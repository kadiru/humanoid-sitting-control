"""Build one-step commands from the measured two-foot balance and rear replant."""
from scipy.spatial.transform import Rotation
import numpy as np
from run import P, read, sha

# Each option uses the same weight transfer and a different lift-and-place target.
OPTIONS=np.array([[0.,0.,0.],[-.036,0,0],[.026,0,0],
                  [-.014,-.028,0],[-.014,.028,0],[-.022,0,-.100],[-.022,0,.100]])
# The donor first crossed 1% just after the transfer. All horizontal adaptation
# takes place later, while the measured shoe is airborne.
GATE=176
BASE_END=431
STAGE_TICKS=650


def ease(t,a,b):
    x=np.clip((t-a)/(b-a),0.,1.)
    return x**3*(10-15*x+6*x*x)


def compile_paths(dest):
    from stance23_planner import chains, legs, forward, solve
    root=P/'compact_maple/evaluation'
    receipt=read(root/'independent_audit.json')
    assert receipt['passed'] and sha(root/'trajectory.npz')==receipt['files']['trajectory.npz']
    result=read(root/'results.json')
    with np.load(root/'trajectory.npz') as d:
        b=d['body'][:BASE_END,0].astype(float)
        b[...,:3]-=np.asarray(result['origins'][0])
        q=d['q'][:BASE_END,0].astype(float)
        command=d['command'][:BASE_END,0,:23].astype(float)
        latent=d['desired'][:BASE_END,0].astype(float)
        load=d['physical_force_w'][:BASE_END,0,5:,2]/d['physical_bodyweight_N'][:BASE_END,0,None]
        slip=d['physical_slip_m_s'][:BASE_END,0,5:]
        prior=d['previous'][0,0,:23].copy()
        assert not d['physical_failure'][:BASE_END,0].any() and not ((load>.01)&(slip>.02)).any()
    ids=[result['body_names'].index(s+'_ankle_roll_link') for s in ('left','right')]
    quat=Rotation.from_quat(b[:,0,3:]).as_matrix()
    feet=b[:,ids,:3]
    terminal_shift=np.median(feet[350:420,0]-feet[0,0],axis=0)[:2]
    paths=[latent.copy()]
    for option in OPTIONS[2:]:
        # Keep the balance transfer identical. Carry the shoe correction into
        # the planted phase so it cannot be commanded to move after contact.
        trajectory=latent.copy()
        for t in range(198,BASE_END):
            reach=ease(t,198,263)
            shift=reach*(option[:2]-terminal_shift)
            yaw=option[2]*reach*1.7
            root_shift=shift*.50*ease(t,345,425)
            for side,(chain,joints) in enumerate(zip(chains,legs)):
                foot,_=forward(chain,q[t,joints])
                R=quat[t]
                goal=foot.copy()
                goal[:3,3]+=R.T@np.r_[(shift if side==0 else 0.)-root_shift,0.]
                if side==0:
                    goal[:3,:3]=R.T@Rotation.from_euler('z',yaw).as_matrix()@R@foot[:3,:3]
                fit,error=solve(chain,q[t,joints],goal,joints)
                assert error<.0015,(option,t,error)
                trajectory[t,joints]+=fit-q[t,joints]
        paths.append(trajectory)
    from rearstep_kin import MassModel
    model=MassModel()
    body_frame=np.empty_like(quat)
    for i,R in enumerate(quat):
        up=R[2]
        f=np.array([1.,0.,0.])-up[0]*up
        f/=np.linalg.norm(f)
        body_frame[i]=np.stack((f,np.cross(up,f),up))
    state=[model.estimate(a,R) for a,R in zip(q,body_frame)]
    np.savez_compressed(dest,reference=np.array(paths,dtype='f'),options=OPTIONS,
                        com=np.stack([v[0] for v in state]),feet=np.stack([v[1] for v in state]),
                        up=body_frame[:,2],weight=load,donor_applied=command,donor_initial_command=prior,
                        donor_trace_sha=receipt['files']['trajectory.npz'])


def floor_witness(env,m):
    """Use the fastest loaded sole points under the existing 2 cm/s slip rule."""
    d=env.robot.data
    loads=(m['force_w'][:,5:,2]/m['bodyweight_N'][:,None]).cpu().numpy()
    speed=m['slip_m_s'][:,5:].cpu().numpy().copy()
    good=np.zeros_like(loads,dtype=bool)
    weight=m['bodyweight_N'].cpu().numpy()
    def arr(x):
        return x.numpy() if hasattr(x,'numpy') else np.asarray(x)
    for side,label in enumerate(('left','right')):
        body=env.robot.body_names.index(label+'_ankle_roll_link')
        s=env.raw.scene['rest_'+label+'_floor']
        vals=s.contact_view.get_contact_data(dt=.005)
        force=np.abs(arr(vals[0]).reshape(-1))
        cp=arr(vals[1]).reshape(-1,3)
        normals=arr(vals[2]).reshape(-1,3)
        counts=arr(vals[4]).reshape(-1)
        starts=arr(vals[5]).reshape(-1)
        velocity=d.body_com_vel_w.torch[:,body].cpu().numpy()
        center=d.body_com_pos_w.torch[:,body].cpu().numpy()
        normal=s.data.normal_force_matrix_w.torch[:,0,0].cpu().numpy()
        active=np.linalg.norm(m['force_w'][:,5+side].cpu().numpy(),axis=1)>.01*weight
        for i in np.flatnonzero(active):
            ix=slice(int(starts[i]),int(starts[i]+counts[i]))
            f=force[ix];pts=cp[ix];n=normals[ix]
            good_points=(f>1.e-7)&np.isfinite(f)&np.isfinite(pts).all(1)&np.isfinite(n).all(1)
            f,pts,n=f[good_points],pts[good_points],n[good_points]
            v=(f[:,None]*n).sum(0)
            mismatch=min(np.linalg.norm(v-normal[i]),np.linalg.norm(v+normal[i]))
            if not len(f) or mismatch>.02+.0003*max(f.sum(),np.linalg.norm(normal[i])):
                continue
            good[i,side]=True
            vel=velocity[i,:3]+np.cross(velocity[i,3:],pts-center[i])
            tangent=vel[:,:2]
            magnitudes=np.linalg.norm(tangent,axis=1)
            order=np.argsort(-magnitudes)
            count=np.searchsorted(np.cumsum(f[order]),.01*weight[i],side='right')
            speed[i,side]=max(magnitudes[order[min(count,len(order)-1)]],
                              np.linalg.norm(np.average(tangent,axis=0,weights=f)))
    active=np.linalg.norm(m['force_w'][:,5:].cpu().numpy(),axis=2)>.01*weight[:,None]
    return loads,speed,good|~active,active
