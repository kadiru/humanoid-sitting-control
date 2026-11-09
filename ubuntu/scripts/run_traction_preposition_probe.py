"""Reposition above the seat before landing; then roll about the pelvis patch.

The first10s are untouched. Static model, encoders, IMU and measured plane are
the only planner inputs. This is a reference-only physical validation probe.
"""
import argparse,sys,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import level_rotation
from crouch_rest_motion import continuous_plan,smooth
from build_near_seat_candidates import solve
from sagittal_static_bias import static_bias
p=argparse.ArgumentParser(add_help=False);p.add_argument('--preposition',type=float,default=.03)
cfg,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_traction_transfer_probe as shear

def plan(model,q,gravity,measurement,reference_origin,command,calibration):
    commands,qs,a=continuous_plan(model,q,gravity,measurement,reference_origin,command,calibration)
    R=level_rotation(gravity);fk=model.fk(q);origin=np.array(a['origin']);n=np.array(a['normal']);d=a['offset'];pitch=a['pitch'];pivot=np.array(a['pivot'])
    feet={s:(R@fk[s+'_ankle_roll_link'][:3,3]+origin,R@fk[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']}
    ids=[model.names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','knee','ankle_pitch']]
    root0=np.r_[origin,Rotation.from_matrix(R).as_quat()];inertial=shear.bridge.inertial
    predicted0,_=static_bias(model,inertial,q,root0,-.2,0.);offset=command[ids]-q[ids]-predicted0
    roots=[];statics=[]
    for k,u in enumerate(np.linspace(0,1,25)):
        pre=smooth(u/.25);land=smooth((u-.25)/.1);roll=smooth((u-.35)/.65)
        angle=pitch+(.1-pitch)*roll
        x=origin[0]+cfg.preposition*pre+((R-Rotation.from_euler('y',angle).as_matrix())@pivot)[0]
        gap=a['gap_initial']+(-.001-a['gap_initial'])*land
        qi,root,error=solve(model,q,feet,n,d,x,angle,gap=gap,y=origin[1]);assert error<.003,(u,error)
        world=Rotation.from_quat(root[3:]).apply(model.vertices['pelvis'])+root[:3];inside=(world[:,0]<-.16)&(world[:,0]>-.64);vertices=world[inside];minimum=(vertices@n+d).min();patch=vertices[vertices@n+d<minimum+.002];sx=float(patch[:,0].mean())
        bias,detail=static_bias(model,inertial,qi,root,sx,smooth((u-.30)/.15))
        commands[k]=command;commands[k,ids]=qi[ids]+bias+(1-smooth(u))*offset;qs[k]=qi;roots.append(root.tolist());statics.append(detail)
    commands[0]=command
    a.update(preposition_config=vars(cfg),root_path=roots,static_path=statics,target_root=roots[-1],first_command_exact=bool(np.array_equal(commands[0],command)))
    return commands,qs,a

class PrepositionEnv(shear.TractionEnv):
    def close(self):
        super().close();(self.out/'preposition_config.json').write_text(json.dumps(vars(cfg),indent=2));(self.out/'source'/'run_traction_preposition_probe.py').write_text(Path(__file__).read_text())
shear.progressive.rolling_plan=plan
shear.progressive.ProgressiveEnv=PrepositionEnv
if __name__=='__main__':shear.progressive.main()
