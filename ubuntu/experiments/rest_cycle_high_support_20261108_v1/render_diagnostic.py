"""Saved evidence only. Rejected FK preview is explicitly NOT simulated motion."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import sys, json, hashlib, subprocess, time, resource
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from PIL import Image, ImageDraw, ImageFont

R = Path('/home/kadir/humanoid-sitting')
E = R/'experiments'
H = Path(__file__).resolve().parent
O = R/'outputs'/H.name/'diagnostic'
sys.path[:0] = [str(R/'sitting_task/src'), str(E/'furniture_saved_review_20261104_v1')]
from humanoid_sitting.shared_frame import BodyModel, level_rotation, apply
from video import paint, group_shapes

W, HEIGHT, FPS = 1280, 840, 10
BG = (245, 247, 250)
INK, RED, GREEN, GRAY = '#183444', '#ac3b32', '#13745a', '#536775'
read = lambda p: json.loads(Path(p).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

@lru_cache(None)
def font(n, bold=False):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf', n)

def text(d, xy, s, n=22, color=INK, bold=False):
    d.text(xy, s, font=font(n, bold), fill=color)

def main():
    start = time.monotonic()
    O.mkdir(exist_ok=False)
    planpath = E/'rest_cycle_material_contact_20261108_v1/model_construction'
    framepath = R/'outputs/rest_cycle_high_support_20261108_v1/sit_high_support/frame_0.npz'
    edgeframe = R/'outputs/rest_cycle_edge_visible_20261108_v1/sit_edge_visible/frame_0.npz'
    files = [planpath/'plan.json', planpath/'complete.json', planpath/'audit.json',
             framepath, edgeframe, H/'preflight.json',
             E/'rest_cycle_front_edge_20261108_v1/model_construction/edge_sensor_association.json',
             E/'furniture_perception_repair_20261029_v1/render_colliders.json',
             R/'runs/g1_kinematics_geometry_04/model.json',
             H/'assets/scenes/high_support_table_050/mesh.npz', Path(__file__)]
    sources = {str(p): sha(p) for p in files}
    audit = read(planpath/'audit.json')
    assert not audit['model_admitted'] and not audit['clearance_audit']['passed']
    plan, landing = read(planpath/'plan.json'), read(planpath/'complete.json')
    nodes = plan['nodes'] + [dict(n, phase='landing') for n in landing['landing'][1:]]
    with np.load(framepath) as z:
        basis = z['root_rotation'] @ level_rotation(z['gravity'].astype(float).copy()).T
        initial = z['root_position'].copy()
    with np.load(edgeframe) as z:
        rgb = z['rgb'][...,:3].astype(np.uint8).copy()
        raw_rgb = Image.fromarray(rgb)
        pixels = tuple(z['selected_pixels'].T)
        rgb[pixels] = .5*rgb[pixels]+.5*np.array([30,230,135])
        selected = Image.fromarray(rgb)
        shade = np.where(np.isfinite(z['depth']), 1-np.clip((z['depth']-.05)/1.5, 0, 1), 0)
        depth = Image.fromarray(np.repeat((255*shade).astype(np.uint8)[...,None], 3, -1))
    with np.load(H/'assets/scenes/high_support_table_050/mesh.npz') as z:
        fv, ff = z['vertices'], np.ascontiguousarray(z['faces'],np.int32)
    shapes = group_shapes(read(E/'furniture_perception_repair_20261029_v1/render_colliders.json')['shapes'])
    model = BodyModel(R/'runs/g1_kinematics_geometry_04/model.json')
    poses = []
    # Same interpolation used in the saved clearance audit; no new audit/solve.
    for a,b in zip(nodes,nodes[1:]):
        ar,br=np.asarray(a['root']),np.asarray(b['root'])
        slerp=Slerp([0,1],Rotation.from_matrix(np.stack([ar[:3,:3],br[:3,:3]])))
        for u in [0.,.25,.5,.75]:
            root=np.eye(4);root[:3,:3]=basis@slerp(u).as_matrix()
            root[:3,3]=initial+basis@((1-u)*ar[:3,3]+u*br[:3,3])
            q=(1-u)*np.asarray(a['q'])+u*np.asarray(b['q'])
            poses.append((root,model.fk(q)))
    root=np.eye(4);last=np.asarray(nodes[-1]['root'])
    root[:3,:3]=basis@last[:3,:3];root[:3,3]=initial+basis@last[:3,3]
    poses.append((root,model.fk(nodes[-1]['q'])))
    assert len(poses)==audit['clearance_audit']['sampled_poses']
    right=np.array([.96,.28,0.]);eye=np.array([.28,-.96,.12]);eye/=np.linalg.norm(eye)
    view=np.stack([right,-np.cross(eye,right),eye],axis=1)
    pw,ph=790,510
    allp=[fv@view]
    for root,fk in poses:
        for name,(v,f,c) in shapes.items():
            allp.append(apply(root@fk[name],v[::max(1,len(v)//30)])@view)
    points=np.concatenate(allp);lo,hi=points.min(0),points.max(0)
    scale=min((pw-50)/(hi[0]-lo[0]),(ph-25)/(hi[1]-lo[1]))
    offset=np.array([pw/2,ph/2,0])-(lo+hi)*scale/2
    def project(v):return np.ascontiguousarray(v@view*scale+offset,np.float32)
    def draw_mesh(rgb,z,v,f,c):
        paint(rgb,z,pw,ph,project(v),np.ascontiguousarray(f,np.int32),
              np.tile(c,(len(f),1)).astype(np.uint8),len(f))
    base=np.full((ph,pw,3),BG,np.uint8);zbase=np.full((ph,pw),-np.inf,np.float32)
    ground=np.array([[-2,-2,0],[2,-2,0],[2,2,0],[-2,2,0]])
    draw_mesh(base,zbase,ground,np.array([[0,1,2],[0,2,3]]),[233,238,243])
    draw_mesh(base,zbase,fv,ff,[181,151,111])
    def scene(index):
        rgb,z=base.copy(),zbase.copy();root,fk=poses[index]
        for name,(v,f,c) in shapes.items():
            color=[177,196,210]
            if name in ('pelvis','left_hip_yaw_link','right_hip_yaw_link'):color=[57,151,137]
            if name in ('left_hip_roll_link','right_hip_roll_link'):color=[219,76,62]
            draw_mesh(rgb,z,apply(root@fk[name],v),f,color)
        return Image.fromarray(rgb)
    first=audit['clearance_audit']['forbidden_or_early_clearance_intersections'][0]['sample']
    edge=read(E/'rest_cycle_front_edge_20261108_v1/model_construction/edge_sensor_association.json')
    discrepancy=abs(np.diff(edge['left_right_center_predictions_m'])[0])*1000
    target=O/'higher_support_validation.mp4'
    cmd=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{HEIGHT}',
         '-r',str(FPS),'-i','-','-an','-c:v','libx264','-threads','1','-preset','fast',
         '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    pipe=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for frame in range(240):
        t=frame/FPS
        im=Image.new('RGB',(W,HEIGHT),BG);d=ImageDraw.Draw(im)
        text(d,(28,20),'Higher table: what passed, and what stopped the sit',29,bold=True)
        text(d,(28,68),'Saved evidence only | 50 cm table | no sitting motion was executed',21,RED)
        if t<5:
            im.paste(scene(0),(8,125));d=ImageDraw.Draw(im)
            text(d,(30,112),'Standing setup reconstruction',23,bold=True)
            im.paste(raw_rgb.resize((384,288)),(867,160));d=ImageDraw.Draw(im)
            text(d,(867,120),'Actual rear-camera image',21,bold=True)
            text(d,(867,480),'PASS: visual/collision mesh',21,GREEN)
            text(d,(867,518),'PASS: depth alignment',21,GREEN)
            text(d,(867,556),'Load-bearing: NOT TESTED',20,RED)
            text(d,(30,675),'The table is present in depth and has matching triangle collisions.',24)
            text(d,(30,718),'These checks do not yet prove it can safely support this robot.',22)
        elif t<17:
            # Visualization of rejected geometry; this is NOT a trajectory replay.
            index=min(len(poses)-1,int((t-5)/12*len(poses)))
            im.paste(scene(index),(8,125));d=ImageDraw.Draw(im)
            text(d,(30,112),'REJECTED PLAN - KINEMATICS ONLY',23,RED,True)
            im.paste(selected.resize((384,288)),(867,160));d=ImageDraw.Draw(im)
            text(d,(867,120),'Current visible support patch',20,bold=True)
            text(d,(867,480),'RED: hip housings',21,RED)
            text(d,(867,518),'TEAL: allowed rest bodies',20,GREEN)
            text(d,(867,558),f'Plan sample {index+1}/{len(poses)}',20)
            status='Clearance violation begins here' if index>=first else 'Moving toward rejected posture'
            text(d,(30,675),status,26,RED if index>=first else INK,True)
            text(d,(30,718),'Hip housings enter the table clearance buffer; this plan was not run.',22)
        else:
            im.paste(selected.resize((576,432)),(38,150))
            im.paste(depth.resize((576,432)),(664,150));d=ImageDraw.Draw(im)
            text(d,(38,115),'Rear RGB + selected plane (green)',22,bold=True)
            text(d,(664,115),'Rear depth - saved startup frame',22,bold=True)
            text(d,(38,620),'The center front edge is hidden by the robot.',25,bold=True)
            text(d,(38,665),f'Two side estimates disagree by {discrepancy:.2f} mm; limit: 3.00 mm.',24,RED)
            text(d,(38,710),'Sit / five-second hold / stand: NOT DEMONSTRATED',27,RED,True)
        text(d,(28,795),'4090 startup captures only | 4070 untouched | unchanged action limits and success criteria',18,GRAY)
        pipe.stdin.write(im.tobytes())
        if frame in [20,120,230]:im.save(O/f'frame_{frame}.jpg',quality=94)
    pipe.stdin.close();assert pipe.wait()==0
    manifest=dict(video=str(target),video_sha256=sha(target),duration_s=24,fps=FPS,frames=240,
        content='Actual saved camera frames plus explicitly labelled rejected kinematic-plan visualization.',
        simulated_motion_video=False,new_simulation_steps=0,new_solver_calls=0,GPU_used=False,
        complete_cycle=False,physical_support_validated=False,sources=sources,
        render_wall_s=time.monotonic()-start,maxrss_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    (O/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='sources'}))

if __name__=='__main__':main()
