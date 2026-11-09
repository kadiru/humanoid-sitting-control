"""Reference-only staged landing with model-based static load allocation."""
import argparse,sys,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

p=argparse.ArgumentParser(add_help=False)
p.add_argument('--settle-fraction',type=float,default=.2)
p.add_argument('--goal-x',type=float,default=-.18)
p.add_argument('--goal-pitch',type=float,default=.1)
p.add_argument('--contact-gap',type=float,default=-.001)
p.add_argument('--contact-ramp',type=float,default=.15)
static_args,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_crouch_rest_handoff as hand
from crouch_rest_motion import continuous_plan,smooth
from build_near_seat_candidates import solve
from sagittal_static_bias import static_bias
from humanoid_sitting.shared_frame import level_rotation
probe=hand.probe
INERTIA=None

def plan(model,q,gravity,measurement,reference_origin,command,calibration):
    commands,qs,a=continuous_plan(model,q,gravity,measurement,reference_origin,command,calibration)
    R=level_rotation(gravity);fk=model.fk(q);origin=np.array(a['origin']);n=np.array(a['normal']);d=a['offset'];pitch=a['pitch']
    feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
    ids=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    root0=np.r_[origin,Rotation.from_matrix(R).as_quat()]
    predicted0,_=static_bias(model,INERTIA,q,root0,-.2,0.)
    offset=command[ids]-q[ids]-predicted0
    statics=[];roots=[]
    for k,u in enumerate(np.linspace(0,1,25)):
        land=smooth(u/static_args.settle_fraction);roll=smooth((u-static_args.settle_fraction)/(1-static_args.settle_fraction))
        angle=pitch+(static_args.goal_pitch-pitch)*roll
        x=origin[0]+(static_args.goal_x-origin[0])*roll
        gap=a['gap_initial']+(static_args.contact_gap-a['gap_initial'])*land
        qi,root,error=solve(model,q,feet,n,d,x,angle,gap=gap,y=origin[1]);assert error<.003,(u,error)
        world=Rotation.from_quat(root[3:]).apply(model.vertices['pelvis'])+root[:3];inside=(world[:,0]<-.16)&(world[:,0]>-.64)
        vertices=world[inside];minimum=(vertices@n+d).min();patch=vertices[vertices@n+d<minimum+.002];sx=float(patch[:,0].mean())
        fraction=smooth((u-static_args.settle_fraction*.5)/static_args.contact_ramp)
        bias,detail=static_bias(model,INERTIA,qi,root,sx,fraction)
        commands[k]=command;commands[k,ids]=qi[ids]+bias+(1-smooth(u))*offset
        qs[k]=qi;roots.append(root.tolist());statics.append(detail)
    commands[0]=command
    a.update(static_config=vars(static_args),root_path=roots,static_path=statics,target_root=roots[-1],first_command_exact=bool(np.array_equal(commands[0],command)),first_knot_jump_rad=float(abs(commands[1]-commands[0]).max()))
    return commands,qs,a

class StaticEnv(hand.HandoffEnv):
    def __init__(self,raw,*args,**kwargs):
        global INERTIA
        INERTIA=json.loads(Path('/data/humanoid-sitting-perception/crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json').read_text())
        super().__init__(raw,*args,**kwargs)
    def close(self):
        super().close();(self.out/'static_config.json').write_text(json.dumps(vars(static_args),indent=2))
        for name in ['run_crouch_rest_statics.py','sagittal_static_bias.py']:(self.out/'source'/name).write_text(Path(__file__).with_name(name).read_text())

probe.progressive.rolling_plan=plan
probe.progressive.ProgressiveEnv=StaticEnv
if __name__=='__main__':probe.progressive.main()
