"""Recorded-pose frustum diagnosis. No renderer, physics or runtime changes.

Truth poses and known finite seat extents are diagnostic labels only. The
unoccluded planar projection is an upper bound, not a synthetic measurement or
proof that the current extractor would accept a rendered depth image.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OUT=D/'visibility_diagnosis_38'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
REAR=np.eye(4);REAR[:3,:3]=np.array([[0.,.707106781,-.707106781],[1.,0.,0.],[0.,-.707106781,-.707106781]]);REAR[:3,3]=[-.07,0,.38]
DOWN=np.eye(4);DOWN[:3,:3]=np.array([[0.,1.,0.],[1.,0.,0.],[0.,0.,-1.]]);DOWN[:3,3]=[-.22,0,.55]

def project(root,K,mount,g):
    R=Rotation.from_quat(root[3:]).as_matrix();C=root[:3]+R@mount[:3,3];RC=R@mount[:3,:3]
    v,u=np.mgrid[:96,:128];rays=np.stack(((u-K[0,2])/K[0,0],(v-K[1,2])/K[1,1],np.ones_like(u)),-1)@RC.T
    top=np.asarray(g['top_center']);n=np.asarray(g['normal']);SR=Rotation.from_quat(g['rotation_xyzw']).as_matrix()
    denom=rays@n
    depth=np.divide((top-C)@n,denom,out=np.full_like(denom,np.nan),where=abs(denom)>1e-8)
    xyz=C+rays*depth[...,None];local=(xyz-top)@SR
    valid=(depth>.08)&(depth<3.)&(abs(local[...,0])<g['depth']/2)&(abs(local[...,1])<g['width']/2)
    # Erode by one axial pixel, mirroring the extractor's normal stencil.
    core=valid[1:-1,1:-1]&valid[1:-1,:-2]&valid[1:-1,2:]&valid[:-2,1:-1]&valid[2:,1:-1]
    points=local[1:-1,1:-1][core]
    span=np.ptp(points[:,:2],axis=0) if len(points) else np.zeros(2)
    return dict(unoccluded_seat_pixels=int(valid.sum()),unoccluded_normal_pixels=int(core.sum()),projected_plane_span_m=span.tolist(),projected_area_upper_bound_m2=float(np.prod(span)),potential_patch=bool(len(points)>=40 and min(span)>=.06 and np.prod(span)>.015)),depth,valid

def main():
    OUT.mkdir(exist_ok=False)
    cap_path=D/'sensor_prefix_v4_24/baseline/raw_sensor_captures.npz';cap=np.load(cap_path)
    base=D/'normal_landing3s_06/candidate';br=read(base/'results.json');j=br['body_names'].index('pelvis')
    body=cap['body'][:,:,j];R=Rotation.from_quat(body[...,3:].reshape(-1,4)).as_matrix().reshape(6,24,3,3)
    pos=body[...,:3]+np.einsum('...ij,j->...i',R,REAR[:3,3])
    quat=Rotation.from_quat(cap['camera_orientation'].reshape(-1,4)).as_matrix().reshape(6,24,3,3)
    pe=float(abs(pos-cap['camera_position']).max());re=float(abs(quat-R@REAR[:3,:3]@np.diag([1,-1,-1])).max())
    assert pe<1e-6 and re<1e-6
    mount_check=dict(passed=True,recorded_samples=144,position_max_component_error_m=pe,rotation_matrix_max_error=re,body_and_camera_quaternion_order='XYZW; raw camera transform is OpenGL, optical ray frame is ROS',source=str(cap_path))
    # An analytic-plane depth comparison uses only old stored pixels, no new render.
    gs=read(base/'geometry.json');origins=np.array(br['origins']);errors=[]
    for k in range(6):
        for i,e in enumerate(br['episodes']):
            root=body[k,i].copy();root[:3]-=origins[i];g=gs[e['geometry_index']]
            info,depth,visible=project(root,cap['K'][k,i],REAR,g);actual=cap['depth'][k,i,:,:,0]
            agreement=visible&np.isfinite(actual)&(abs(actual-depth)<.01)
            if agreement.any():errors.extend(abs(actual[agreement]-depth[agreement]).tolist())
    depth_check=dict(agreement_pixels=len(errors),median_depth_error_m=float(np.median(errors)) if errors else None,p99_depth_error_m=float(np.quantile(errors,.99)) if errors else None,scope='Only pixels within10mm of the independently predicted seat plane are counted; this does not verify all occlusions or prove exact rendered-depth reconstruction.')
    P=D/'staged_approach_36/candidate';r=read(P/'results.json');tr=np.load(P/'trace.npz');gs=read(P/'geometry.json');origins=np.array(r['origins']);j=r['body_names'].index('pelvis')
    K=cap['K'][0,0];KD=np.array([[128*18/24,0,64.],[0,128*18/24,48.],[0,0,1.]])
    rows=[]
    for i in [0,1,2,3,6,7,10,11]:
        samples=[];g=gs[r['episodes'][i]['geometry_index']]
        for k,time in enumerate(tr['t']):
            root=tr['body'][k,i,j].copy();root[:3]-=origins[i]
            rear,_,_=project(root,K,REAR,g);down,_,_=project(root,KD,DOWN,g)
            samples.append(dict(time_s=float(time),recorded_geometry_valid=bool(tr['geometry'][k,i,11]),recorded_geometry_age_s=float(tr['geometry'][k,i,12]),root_pitch_rad=float(Rotation.from_quat(root[3:]).as_euler('xyz')[1]),rear=rear,existing_down_mount_hypothesis=down))
        row=dict(case=i,height_cm=g['height']*100,slope_deg=g['pitch_deg'],samples=samples);rows.append(row)
        print(i,[(s['time_s'],s['recorded_geometry_valid'],s['rear']['unoccluded_normal_pixels'],s['rear']['potential_patch'],s['existing_down_mount_hypothesis']['unoccluded_normal_pixels'],s['existing_down_mount_hypothesis']['potential_patch']) for s in samples if s['time_s'] in [8.,10.,12.,14.,16.,20.]],flush=True)
    fig,axs=plt.subplots(2,4,figsize=(15,6),sharex=True)
    for col,row in enumerate([rows[k] for k in [0,2,4,6]]):
        ss=row['samples'];time=[x['time_s'] for x in ss]
        axs[0,col].plot(time,[x['rear']['unoccluded_normal_pixels'] for x in ss],label='Active rear camera')
        axs[0,col].plot(time,[x['existing_down_mount_hypothesis']['unoccluded_normal_pixels'] for x in ss],'--',label='Disabled down-camera hypothesis')
        axs[1,col].step(time,[x['recorded_geometry_valid'] for x in ss],where='post',label='Recorded usable plane')
        axs[1,col].step(time,[x['rear']['potential_patch'] for x in ss],where='post',label='Rear frustum upper bound')
        axs[0,col].set_title(f"{row['height_cm']:g}cm / +{row['slope_deg']:g}°");axs[1,col].set_xlabel('Episode time (s)');axs[1,col].set_ylim(-.1,1.1)
        for ax in axs[:,col]:ax.grid(alpha=.2)
    axs[0,0].set_ylabel('Unoccluded plane-normal pixels');axs[1,0].set_ylabel('Usable / potentially visible');axs[0,0].legend(fontsize=8);axs[1,0].legend(fontsize=8)
    fig.suptitle('Recorded-pose camera coverage diagnosis — no new rendering or physics\nUnoccluded upper bound; down-camera source is disabled in the current controller');fig.tight_layout();fig.savefig(OUT/'visibility.png',dpi=160);plt.close(fig)
    paths=[Path(__file__),cap_path,base/'results.json',P/'trace.npz',P/'geometry.json',P/'results.json',Path('scripts/run_supported_postures.py'),Path('scripts/run_resting_posture.py')]
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Diagnostic labels only. Existing stored body poses/known finite surfaces and verified camera mount yield an unoccluded frustum upper bound. Does NOT replace missing measurements, emulate self-occlusion or prove current plane fitting would succeed. Down camera is a hypothesis using an existing code-defined mount that is disabled in this study. No runtime/camera/controller changes, no rendering/physics/PPO/holdout.',mount_check=mount_check,old_pixel_plane_check=depth_check,physics_ticks=0,ppo_updates=0,source_hashes={str(p.resolve()):sha(p) for p in paths},rows=rows)
    (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
