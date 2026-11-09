"""A common 3D frame built only from saved depth, encoders, gravity and calibration."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import ConvexHull
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from humanoid_sitting.shared_frame import BodyModel,unproject,transform,apply,level_rotation,voxel
from seat_depth_estimator import estimate_body_frame

p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
root=Path(__file__).resolve().parents[1]
data=np.load('/data/humanoid-sitting-perception/support_rl_camera_probe_02/startup_capture.npz')
model=BodyModel(root/'runs/g1_kinematics_geometry_04/model.json')
T=transform(Rotation.from_quat(data['quat']).as_matrix(),data['mount'])
fig=plt.figure(figsize=(13,7));fig.subplots_adjust(left=.01,right=.99,top=.80,bottom=.08,wspace=.03)
records=[]
for column,case in enumerate((0,8)):
    depth,K,gravity,q=[data[key][case] for key in ('depth','K','gravity','q')]
    estimate=estimate_body_frame(depth,K,data['mount'],data['quat'],gravity)
    R=level_rotation(gravity);poses=model.fk(q)
    points=unproject(depth,K,T);points=points[model.self_mask(points,poses)]
    scene=voxel(points,.022)@R.T;scene[:,2]-=estimate['floor_height_body_m']
    keep=(scene[:,0]>-1)&(scene[:,0]<.35)&(abs(scene[:,1])<.65)&(scene[:,2]<1.1)
    scene=scene[keep]
    seat_z=estimate['surface_center'][2]
    floor=abs(scene[:,2])<.018;seat=abs(scene[:,2]-seat_z)<.012
    ax=fig.add_subplot(1,2,column+1,projection='3d')
    for mask,color,label in ((~(floor|seat),'#94a3b8','Other observed points'),(floor,'#64748b','Observed floor'),(seat,'#059669','Observed seat')):
        ax.scatter(*scene[mask].T,c=color,s=3,alpha=.6,label=label)
    skeleton=model.skeleton(poses)@R.T;skeleton[:,:,2]-=estimate['floor_height_body_m']
    for j,segment in enumerate(skeleton):ax.plot(*segment.T,color='#d97706',lw=3,label='Body from joint angles' if j==0 else None)
    pelvis=model.vertices['pelvis']@R.T;pelvis[:,2]-=estimate['floor_height_body_m']
    ax.add_collection3d(Poly3DCollection(pelvis[ConvexHull(pelvis).simplices],facecolor='#f59e0b',edgecolor='none',alpha=.18))
    polygon=np.asarray(estimate['polygon']);polygon=np.c_[polygon,np.full(len(polygon),seat_z)]
    ax.add_collection3d(Poly3DCollection([polygon],facecolor='#10b981',edgecolor='#047857',alpha=.18))
    clearance=float(pelvis[:,2].min()-seat_z)
    ax.set(title=f'Start {case}: measured seat height {seat_z*100:.1f} cm\nPelvis bottom to seat plane: {clearance*100:.1f} cm',
        xlim=(-1,.4),ylim=(-.65,.65),zlim=(0,1.4),xlabel='Forward / backward (m)',ylabel='Sideways (m)',zlabel='Height above detected floor (m)')
    ax.set_xticks([-.8,-.4,0]);ax.set_yticks([-.4,0,.4]);ax.set_zticks([0,.4,.8,1.2])
    ax.view_init(elev=19,azim=-61);ax.set_box_aspect((1.4,1.3,1.4))
    if column==0:ax.legend(loc='upper left',bbox_to_anchor=(.01,1.0),fontsize=8)
    records.append(dict(case=case,estimated_seat_height_m=seat_z,vertical_pelvis_clearance_m=clearance,visible_scene_points=len(scene)))
fig.suptitle('Robot, floor and seat in one measured 3D frame\nDepth camera + joint angles + gravity; no simulator pose or seat dimensions',fontsize=14)
fig.savefig(out/'common_sensor_frame.png',dpi=160);plt.close(fig)
(out/'metadata.json').write_text(json.dumps(dict(inputs=['depth','camera intrinsics/extrinsics','encoders','gravity','static robot kinematics/collision geometry'],
    simulator_truth_used=False,frame='Gravity aligned; detected floor z=0; XY origin at current pelvis',
    seat_polygon='Visible patch only; not complete hidden furniture bounds',measurements=records),indent=2)+'\n')
print(json.dumps(records))
