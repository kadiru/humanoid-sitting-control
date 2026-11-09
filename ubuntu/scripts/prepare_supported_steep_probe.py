"""Generate kinematic starting guesses. Physics validation decides admissibility."""
import argparse,json,sys,shutil
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'sitting_task/src'))
from humanoid_sitting.shared_frame import BodyModel,apply
from humanoid_sitting.supported_surface import surface_frame

p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--only-kind',choices=['sit','lean']);a=p.parse_args()
root=Path(__file__).resolve().parents[1];out=Path(a.output);out.mkdir(exist_ok=False)
m=BodyModel(root/'runs/g1_kinematics_geometry_04/model.json');data=json.loads((root/'runs/g1_kinematics_geometry_04/model.json').read_text())
old=np.load('/data/humanoid-sitting-perception/resting_seated_capture_dev02/candidates.npz')
items=[];qs=[];roots=[];commands=[]
def add(q,root_R,command,pitch,kind,label,foot_x=0.,gap=.002):
    if a.only_kind is not None and a.only_kind!=kind:return
    q=q.copy();command=command.copy()
    fk=m.fk(q)
    feet=np.array([fk[s+'_ankle_roll_link'][:3,3]@root_R.T for s in ('left','right')])
    soles=np.concatenate([apply(fk[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link'])@root_R.T for s in ('left','right')])
    pos=np.array([foot_x-feet[:,0].mean(),-feet[:,1].mean(),.002-soles[:,2].min()])
    points={name:apply(T,m.vertices[name])@root_R.T+pos for name,T in fk.items() if name in m.vertices}
    R=surface_frame(pitch);n=R[:,2];u=R[:,0]
    allowed=['pelvis','left_hip_yaw_link','right_hip_yaw_link'] if kind=='sit' else ['pelvis','torso_link']
    # Put the plane just behind the intended contact patch; finite extent matters.
    proxy=np.concatenate([points[x] for x in allowed])
    depth=.50 if kind=='sit' else .48;width=.85
    center=pos-.17*u if kind=='sit' else pos+root_R@np.array([0.,0.,.04])
    # Fit the plane to the finite patch, not to a thigh vertex outside the seat.
    if kind=='sit':proxy=points['pelvis']
    uv=(proxy-center)@R
    subset=proxy[(abs(uv[:,0])<depth/2-.02)&(abs(uv[:,1])<width/2-.02)]
    if not len(subset):raise ValueError('No intended contact patch within support bounds')
    center+=n*((subset@n).min()-gap-center@n)
    nearest=[]
    for name,pts in points.items():
        uv=(pts-center)@R;inside=(abs(uv[:,0])<depth/2)&(abs(uv[:,1])<width/2)
        if inside.any():nearest.append((float(uv[inside,2].min()),name))
    forbidden=[(d,b) for d,b in nearest if b not in allowed and d<-.003 and 'ankle_roll' not in b]
    item=dict(case=len(items),label=label,kind=kind,pitch_deg=pitch,depth=depth,width=width,
        height=float(center[2]),x=float(center[0]),y=float(center[1]),yaw=0.,family='slab',
        top_center=center.tolist(),rotation_xyzw=Rotation.from_matrix(R).as_quat().tolist(),
        normal=n.tolist(),initial_root_pitch_deg=float(Rotation.from_matrix(root_R).as_euler('xyz',degrees=True)[1]),
        initial_forbidden_penetrations=forbidden,nearest_bodies=sorted(nearest)[:8])
    items.append(item);qs.append(q);commands.append(command);roots.append(np.r_[pos,Rotation.from_matrix(root_R).as_quat()])

for index,height in zip([0,4,8,12],[.25,.30,.35,.40]):
    q=old['q'][index];R=Rotation.from_quat(old['root_pose'][index,3:]).as_matrix()
    for angle in [0.,10.,20.,30.,45.]:
        for gap in [.001,.004]:add(q,R,old['command'][index],angle,'sit',f'seated_{height}_{angle}_{gap}',gap=gap)
for knee in [.42,.70,1.]:
    for tilt in [-16.,-20.,-24.]:
        q=np.array(data['default_q']);R=Rotation.from_euler('y',tilt,degrees=True).as_matrix()
        for side in ('left','right'):
            q[m.names.index(side+'_hip_pitch_joint')]=-.5*knee
            q[m.names.index(side+'_knee_joint')]=knee
            q[m.names.index(side+'_shoulder_pitch_joint')]=-.8
            q[m.names.index(side+'_shoulder_roll_joint')]=.3 if side=='left' else -.3
            q[m.names.index(side+'_elbow_pitch_joint')]=1.2
        fk=m.fk(q)
        for side in ('left','right'):
            angle=Rotation.from_matrix(R@fk[side+'_ankle_roll_link'][:3,:3]).as_euler('xyz')[1]
            q[m.names.index(side+'_ankle_pitch_joint')]-=angle
        for angle in [75.,90.]:add(q,R,q,angle,'lean',f'lean_k{knee}_tilt{tilt}_surface{angle}')
np.savez_compressed(out/'candidates.npz',q=np.array(qs,np.float32),qd=np.zeros_like(qs,dtype=np.float32),
    command=np.array(commands,np.float32),root_pose=np.array(roots,np.float32),root_velocity=np.zeros((len(items),6),np.float32))
(out/'geometry.json').write_text(json.dumps(items,indent=2)+'\n')
shutil.copy2(Path(__file__),out/'prepare_source.py')
print(json.dumps(dict(candidates=len(items),kinematic_intersection_flags=sum(bool(x['initial_forbidden_penetrations']) for x in items),by_kind={k:sum(x['kind']==k for x in items) for k in ('sit','lean')})))
