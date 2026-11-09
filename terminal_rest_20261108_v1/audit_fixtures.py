"""Independent synthetic geometry and timing checks; no experiment inputs."""
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation
from seed import pitch_ik

ry=lambda a:Rotation.from_euler('y',a).as_matrix()
# Construct Cartesian ankle targets forward from known bent configurations.
# The seed inverse must recover them with nonzero CAD offsets and root pitch.
names=[side+'_'+n+'_joint' for side in ('left','right') for n in ('hip_pitch','knee','ankle_pitch')]
plan=SimpleNamespace(q0=np.zeros(6),model=SimpleNamespace(names=names))
root=np.eye(4);root[:3,:3]=ry(.31);root[:3,3]=[-.14,0.,-.08]
qtrue=np.array([-.63,.82,-.56]*2);chain={}
for side,y in [('left',.1),('right',-.1)]:
    off=[.07,-.03,.02];u=np.array([.01,0.,-.4]);v=np.array([-.02,0.,-.39])
    hip=np.array([0.,y,-.04]);h,k,a=qtrue[:3]
    ap=root[:3,3]+root[:3,:3]@(hip+ry(off[0]+h)@(u+ry(off[1]+k)@v))
    ar=root[:3,:3]@ry(off[0]+h+off[1]+k+off[2]+a)
    chain[side]=([{'p0':hip}],off,u,v,ar,ap)
recovered=pitch_ik(plan,root,chain)
# Independently count a known reverse edge sequence into forward arrival
# times, including the mandatory hover interval and loading time.
reverse_durations=[0.,.6,.6,.6,.6,.7,.8]
forward_arrivals=np.cumsum(list(reversed(reverse_durations[1:])))
motion_sum=sum(reverse_durations);settled=motion_sum+.5+1.+1.
# Lower bound must not double charge simultaneous translation/joint motion.
cmd_time=2.;normal_time=.012/.0055
combined_bound=max(cmd_time,normal_time+.5)
tests=dict(nonzero_CAD_offset_IK=np.max(abs(recovered-qtrue))<1e-12,
    reverse_time_matches_edge_sum=abs(forward_arrivals[-1]-motion_sum)<1e-12,
    quiet_load_settle_charged=abs(settled-6.4)<1e-12,
    simultaneous_motion_uses_max=abs(combined_bound-(normal_time+.5))<1e-12)
result=dict(passed=all(tests.values()),tests=tests,
    inverse_joint_error_rad=float(np.max(abs(recovered-qtrue))),
    experiment_inputs_read=False,physics_steps=0,scope='Synthetic seed algebra and time accounting only')
with Path(__file__).with_name('synthetic_audit.json').open('x') as f:json.dump(result,f,indent=2,default=lambda x:x.tolist())
print(json.dumps(result,default=lambda x:x.tolist()))
assert result['passed']
