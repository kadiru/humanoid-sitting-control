"""CPU raster replay of saved native poses and forces, never simulation."""
import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
import time
START=time.monotonic()
import json,sys,subprocess,hashlib,ctypes,resource
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image,ImageDraw,ImageFont


def main():
    c=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(exist_ok=False)
    for p,h in c['sha256'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
    with np.load(c['trajectory']) as archive:z={k:archive[k] for k in archive.files}
    poses=z['pose'];forces=z['normal']+z['friction'];names=list(z['link_names'])
    mass=z['mass'][0].astype(float);weight=mass.sum()*9.81
    shapes=json.loads(Path(c['colliders']).read_text())['shapes'];bodies={}
    for shape in shapes:bodies.setdefault(shape['body'],[]).append(shape)
    grouped={}
    for name,items in bodies.items():
        vertices=[];faces=[];offset=0
        for item in items:
            vertices.extend(item['vertices']);faces.extend(np.asarray(item['faces'])+offset);offset+=len(item['vertices'])
        grouped[name]=(np.asarray(vertices),np.asarray(faces,dtype=np.int32))
    paint=ctypes.CDLL(c['raster']).paint;arr=np.ctypeslib.ndpointer
    paint.argtypes=[arr(np.uint8,flags='C'),arr(np.float32,flags='C'),ctypes.c_int,ctypes.c_int,
        arr(np.float32,flags='C'),arr(np.int32,flags='C'),arr(np.uint8,flags='C'),ctypes.c_int]
    paint.restype=None
    cube=np.array([[x,y,z] for x in [-.5,.5] for y in [-.5,.5] for z in [-.5,.5]])
    from scipy.spatial import ConvexHull
    faces=np.ascontiguousarray(ConvexHull(cube).simplices,np.int32)
    support=c['support'];rs=Rotation.from_quat(support['quaternion_xyzw']).as_matrix()
    seat=(cube*np.asarray(support['size']))@rs.T+support['position']
    floor=cube*np.array([1.8,1.4,.03])+[0,0,-.015]
    right=np.array([.93,.3675595,0]);right/=np.linalg.norm(right)
    eye=np.array([right[1]*.98,-right[0]*.98,.2]);eye/=np.linalg.norm(eye)
    view=np.stack([right,-np.cross(eye,right),eye],axis=1)
    W,H=1280,760;pw,ph=880,560;scale=460.;center=np.array([-.08,0,.48])@view
    offset=np.array([pw/2,ph/2,0])-center*scale
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    fonts={n:ImageFont.truetype(font_path,n) for n in (17,20,23,27,32)}
    def txt(draw,xy,text,size=23,color='#183444'):draw.text(xy,text,font=fonts[size],fill=color)
    target=out/'cpu_unloading_failure.mp4'
    pipe=subprocess.Popen(['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}',
        '-r','10','-i','-','-an','-c:v','libx264','-threads','1','-preset','fast','-crf','20',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE)
    events=[]
    try:
        for number,k in enumerate(range(19,len(poses),20)):
            if time.monotonic()-START>42:raise TimeoutError('REPLAY_45S_FINITE_RETURN')
            image=np.full((ph,pw,3),[244,247,249],np.uint8);depth=np.full((ph,pw),-np.inf,np.float32)
            def mesh(v,f,color):
                projected=np.ascontiguousarray(v@view*scale+offset,np.float32)
                colors=np.tile(color,(len(f),1)).astype(np.uint8)
                paint(image,depth,pw,ph,projected,np.ascontiguousarray(f,np.int32),colors,len(f))
            mesh(floor,faces,[210,219,222]);mesh(seat,faces,[185,147,97])
            rotations=Rotation.from_quat(poses[k,:,3:]).as_matrix()
            for name,(v,f) in grouped.items():
                idx=names.index(name);color=[116,163,190]
                if name=='pelvis':color=[27,145,116]
                if 'hip_' in name:color=[217,124,67]
                mesh(v@rotations[idx].T+poses[k,idx,:3],f,color)
            im=Image.new('RGB',(W,H),'#f4f7f9');im.paste(Image.fromarray(image),(10,120));draw=ImageDraw.Draw(im)
            sec=(k+1)*.005;seat_load=forces[k,:,1].sum(0)@support['normal']/weight
            ground=forces[k,:,0,2].sum()/weight
            tilt=np.rad2deg(np.arccos(np.clip(rotations[0,2,2],-1,1)))
            speed=float(abs(z['qd'][k,:23]).max())
            txt(draw,(25,18),'CPU unloading test — unsuccessful return',32)
            txt(draw,(25,66),f'Actual saved physics poses | starts already seated | {sec:.1f} / 25.0 s',20)
            txt(draw,(910,142),'Measured forces',27)
            txt(draw,(910,198),f'Seat: {100*seat_load:.1f}% weight')
            txt(draw,(910,240),f'Floor: {100*ground:.1f}% weight')
            txt(draw,(910,307),f'Body tilt: {tilt:.1f} deg')
            txt(draw,(910,350),f'Joint speed: {speed:.3f}',20)
            txt(draw,(910,380),'rad/s (native reading)',17)
            active=[names[i] for i,f in enumerate(forces[k,:,1]) if np.linalg.norm(f)>1]
            thighs=any('hip_yaw' in name for name in active)
            txt(draw,(910,443),'Seat contact:',23)
            txt(draw,(910,481),'pelvis + thighs' if thighs and 'pelvis' in active else 'thighs' if thighs else 'pelvis' if 'pelvis' in active else 'none',23)
            if sec<=5:status='Seated measurement baseline; not a complete sitting cycle.'
            elif sec<14:status='Reversed commands move weight toward the feet.'
            elif sec<16:status='The seat unloads briefly, but balance is not maintained.'
            else:status='The robot tips backward; thigh contact invalidates the simple load model.'
            txt(draw,(25,690),status,20,'#a0372d' if sec>=14 else '#183444')
            txt(draw,(25,730),'CPU raster replay; no GPU, camera image, new simulation, or successful cycle claimed.',17)
            pipe.stdin.write(im.tobytes())
            if number in [49,149,199,249]:im.save(out/f'frame_{number}.jpg',quality=94)
            events.append(dict(time_s=sec,seat_bw=float(seat_load),floor_bw=float(ground),tilt_deg=float(tilt),seat_bodies=active))
        pipe.stdin.close();assert pipe.wait(timeout=3)==0
    finally:
        if pipe.poll() is None:pipe.kill();pipe.wait()
    result=dict(duration_s=len(events)/10,frames=len(events),fps=10,playback_speed=1.,
        saved_native_poses=True,visual_mesh_source=c['colliders'],no_camera=True,GPU_contexts=0,
        new_physics_steps=0,complete_cycles=0,physical_success=False,wall_s=time.monotonic()-START,
        process_CPU_s=time.process_time(),sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (out/'manifest.json').write_text(json.dumps(result,indent=2));(out/'frame_signals.json').write_text(json.dumps(events))
    print(json.dumps(result))


if __name__=='__main__':main()
