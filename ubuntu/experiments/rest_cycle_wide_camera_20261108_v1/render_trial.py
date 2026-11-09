"""Complete20s replay of saved physical poses, at1x; no simulation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,json,hashlib,subprocess,functools
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent
OUT=R/'outputs'/H.name;TRIAL=OUT/'sit_coffee_wide'
sys.path.insert(0,str(R/'experiments/furniture_collision_review_20261106_v1'))
from render import Scene
W,HEIGHT,FPS=1280,900,12.5
BG='#f5f7fa';INK='#183444';RED='#ad3a30';GREEN='#14765b'
@functools.lru_cache(None)
def font(size):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',size)
def txt(d,xy,s,size=23,color=INK):d.text(xy,s,font=font(size),fill=color)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    audit=json.loads((TRIAL/'independent_audit.json').read_text())
    signals=json.loads((TRIAL/'controller_signals.json').read_text())
    with np.load(TRIAL/'trajectory.npz') as f:z=dict(f)
    assert len(z['body'])==1000 and len(signals)==1000
    scene=Scene(dict(id='sit_modern_coffee_table_01_center',source=str(TRIAL),full20s=True))
    snapshots={}
    for tick in [0,350,650,1000]:
        with np.load(TRIAL/f'frame_{tick}.npz') as f:snapshots[tick]=Image.fromarray(f['rgb'][...,:3].astype(np.uint8))
    target=OUT/'wide_camera_sitting_trial.mp4'
    assert not target.exists()
    cmd=['ffmpeg','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{HEIGHT}','-r','25/2','-i','-',
         '-an','-c:v','libx264','-threads','1','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(target)]
    pipe=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    for frame in range(250):
        tick=4*frame+3;time=(tick+1)*.02
        im=Image.new('RGB',(W,HEIGHT),BG);d=ImageDraw.Draw(im)
        txt(d,(28,18),'Sitting trial: wider rear-camera view',32)
        txt(d,(28,65),f'Saved simulation poses | normal speed | {time:.2f} /20.00s',21)
        im.paste(scene.frame(tick),(8,110))
        snap=max(k for k in snapshots if k<=tick+1)
        im.paste(snapshots[snap].resize((336,252)),(924,154));d=ImageDraw.Draw(im)
        txt(d,(924,119),'Rear camera',22)
        txt(d,(924,422),f'Snapshot at {snap*.02:.2f}s',20)
        sg=signals[tick]
        txt(d,(924,475),f"Stage: {sg['phase']}",21)
        txt(d,(924,515),f"Camera age: {sg['geometry_age_s']:.2f}s",20)
        txt(d,(924,555),f"Quiet: {'yes' if sg['quiet'] else 'no'}",20)
        hold=float(scene.hold[tick]);load=float(z['support_bw'][tick,0])
        txt(d,(28,700),f'Qualifying hold: {hold:.2f} /5.00s    Reported support: {100*load:.1f}% bodyweight',25)
        issues=[]
        if scene.failure[tick]:issues.append('physical failure')
        if scene.slip[tick]:issues.append('loaded slip')
        if not scene.legal[tick]:issues.append('forbidden contact')
        txt(d,(28,747),'Safety record so far: '+(', '.join(issues) if issues else 'no violation recorded'),24,RED if issues else GREEN)
        if frame==249:
            txt(d,(28,798),f"Final result: {'PASS' if audit['admitted'] else 'FAIL'} | final hold {audit['final_hold_s']:.2f}s",27,GREEN if audit['admitted'] else RED)
        else:txt(d,(28,798),'Geometry-guided controller; this is not PPO and does not include standing up.',21)
        txt(d,(28,857),'Unchanged action limits and physical criteria |4090 only | reconstructed from saved measurements',18)
        pipe.stdin.write(im.tobytes())
        if frame in [86,161,249]:im.save(OUT/f'video_frame_{frame}.jpg',quality=94)
    pipe.stdin.close();assert pipe.wait()==0
    manifest=dict(path=str(target),sha256=sha(target),source_trajectory_sha256=sha(TRIAL/'trajectory.npz'),
        renderer_sha256=sha(Path(__file__)),duration_s=20,frames=250,fps=12.5,playback_speed=1.,
        source='saved full20s physical body poses, camera snapshots, and original criteria; no resimulation',
        physical_pass=bool(audit['admitted']),final_hold_s=audit['final_hold_s'],complete_cycle=False,
        GPU_used=False,PPO=False)
    with (OUT/'video_manifest.json').open('x') as f:json.dump(manifest,f,indent=2)
    print(json.dumps(manifest))

if __name__=='__main__':main()
