"""53-second full-duration tiled replay of recorded physics; no simulation."""
import functools
import subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
from io_utils import OUT,ASSETS,read,write,sha

W,H=2560,1440
CW,CH=628,310
FPS=10
BG='#edf2f5';INK='#233b4d';RED='#bf403e';GREEN='#167657';AMBER='#b66b1f'

@functools.lru_cache(None)
def font(size,bold=False):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf',size)

def main():
    cases=read(ASSETS/'cases.json');result=read(OUT/'probes/result.json');data=read(OUT/'probes/measurements.json')
    cache={};trials={}
    view=np.array([[.8,.6,0.],[-.18,.24,.954]])
    depth=np.cross(view[0],view[1])
    for c in cases:
        z=np.load(ASSETS/'scenes'/c['id']/'mesh.npz');vertices=z['vertices'];faces=z['faces']
        corners=np.vstack([vertices,[[.5,-.35,0],[.5,.35,1.35]]]);uv=corners@view.T
        scale=min((CW-42)/(uv[:,0].max()-uv[:,0].min()),185/(uv[:,1].max()-uv[:,1].min()))
        offset=np.array([CW/2-scale*(uv[:,0].max()+uv[:,0].min())/2,255+scale*uv[:,1].min()])
        def proj(v,sc=scale,of=offset):
            u=np.asarray(v)@view.T;return np.stack([of[0]+sc*u[...,0],of[1]-sc*u[...,1]],-1)
        im=Image.new('RGB',(CW,CH),'white');d=ImageDraw.Draw(im)
        tri=vertices[faces];order=np.argsort(tri.mean(1)@depth)
        normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normals/=np.maximum(np.linalg.norm(normals,axis=1,keepdims=True),1e-20)
        light=np.array([.5,-.3,.8]);light/=np.linalg.norm(light)
        for j in order:
            shade=.72+.24*max(float(normals[j]@light),0.)
            color=tuple(int(x*shade) for x in [198,185,157])
            d.polygon([tuple(v) for v in proj(tri[j])],fill=color,outline='#b5aca0',width=1)
        cache[c['id']]=(im,proj)
        folder=OUT/('robot_'+c['id'])
        if (folder/'independent_audit.json').exists():
            z=np.load(folder/'trajectory.npz')
            trials[c['id']]=dict(trace={k:z[k] for k in ['body','force','support_bw','good','alive']},
                mapping=read(folder/'contact_mapping.json'),audit=read(folder/'independent_audit.json'),
                results=read(folder/'results.json'),shapes=read(folder/'robot_collision_shapes.json')['shapes'])

    def arrow(d,proj,p,f,scale):
        p=np.asarray(p);f=np.asarray(f)
        if not np.isfinite(p).all() or np.linalg.norm(f)<.1:return
        a,b=proj(np.array([p,p+scale*f]));d.line([tuple(a),tuple(b)],fill=RED,width=4)
        delta=b-a;length=np.linalg.norm(delta)
        if length>4:
            u=delta/length;v=np.array([-u[1],u[0]])
            d.polygon([tuple(b),tuple(b-9*u+4*v),tuple(b-9*u-4*v)],fill=RED)

    def card(c,phase,tick):
        im,proj=cache[c['id']];im=im.copy();d=ImageDraw.Draw(im)
        title=c['title']+' / '+c['approach'].replace('_',' ')
        d.text((12,8),title[:48],font=font(18,True),fill=INK)
        if phase=='probe':
            round_id,t=divmod(tick,60);r=data[round_id][t*5][cases.index(c)]
            x,y=proj(r['position']);d.ellipse((x-5,y-5,x+5,y+5),fill='#3089a2',outline='#16465c',width=2)
            cp=np.mean(r['points'],0) if r['points'] else r['position']
            arrow(d,proj,cp,r['force'],.02)
            row=next(x for x in result['checks'] if x['case']==c['id'] and x['round']==round_id)
            d.text((12,34),('PRESS '+str(round_id+1) if round_id<3 else 'EMPTY-SPACE CONTROL')+
                ' | '+('PASS' if row['passed'] else 'FAIL'),font=font(17,True),fill=GREEN if row['passed'] else RED)
            d.text((12,275),f"t={r['t']:4.2f}s  contact {np.linalg.norm(r['force']):5.2f} N  target {'yes' if r['labels'] and all(x['valid'] for x in r['labels']) else 'no'}",font=font(16),fill=INK)
        else:
            trial=trials.get(c['id'])
            if trial is None:
                d.text((12,34),'ROBOT TEST SKIPPED: collision check failed',font=font(17,True),fill=RED)
                d.text((12,275),'No robot success claimed.',font=font(16),fill=INK)
            else:
                k=tick*5;pose=trial['trace']['body'][k,0];origin=np.array(trial['results']['origins'][0]);names=trial['results']['body_names']
                shapes=[]
                for s in trial['shapes']:
                    b=names.index(s['body']);rot=Rotation.from_quat(pose[b,3:7]).as_matrix()
                    v=np.array(s['vertices'])@rot.T+pose[b,:3]-origin;tri=v[np.array(s['faces'])]
                    for q in tri:shapes.append((float(q.mean(0)@depth),q))
                for _,q in sorted(shapes,key=lambda x:x[0]):d.polygon([tuple(v) for v in proj(q)],fill='#92b8cc',outline='#46748e',width=1)
                contacts=trial['mapping']['frames'][25+k]['points']
                for j,point in enumerate(contacts):
                    if point is not None:arrow(d,proj,np.array(point)-origin,trial['trace']['force'][k,0,j],.002)
                a=trial['audit'];label='UNVALIDATED' if a['clean'] is None else 'PASS' if a['clean'] else 'FAIL'
                d.text((12,34),f"FULL ROBOT TRIAL: {label} | {c['task']}",font=font(17,True),fill=GREEN if a['clean'] else RED)
                d.text((12,275),f"t={(k+1)*.02:4.2f}s  support {100*trial['trace']['support_bw'][k,0]:4.1f}% BW  hold {a['final_hold_s']:.2f}s",font=font(16),fill=INK)
        d.rectangle((1,1,CW-2,CH-2),outline='#becbd3',width=2)
        return im

    def grid(phase,tick):
        im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
        title='Furniture collision checks: calibrated physical probes' if phase=='probe' else 'Autonomous robot tests: entire20-second attempts'
        d.text((22,14),title,font=font(32,True),fill=INK)
        d.text((22,57),'Recorded simulator states,1x speed. Mesh lines: configured collision shapes. Red arrows: measured contact force.',font=font(22),fill=INK)
        d.text((22,90),'Probe arrow scale:2cm/N. Robot arrow scale:2mm/N. No synthetic motion or forces. Scores use every20ms sample.',font=font(20),fill=INK)
        for i,c in enumerate(cases):im.paste(card(c,phase,tick),(14+(CW+8)*(i%4),135+(CH+8)*(i//4)))
        x=14+2*(CW+8);y=135+3*(CH+8)
        lines=['UNCHANGED ROBOT SUCCESS','Full20s; final5s supported and quiet.','No loaded slip above2cm/s.','No forbidden contact, hard impact,','fall, torque or joint-limit violation.','Probe PASS is not robot success.']
        if phase=='robot':lines+=['All14 are new furniture tests,','not repairs of the protected72.']
        for i,line in enumerate(lines):d.text((x+12,y+12+i*30),line,font=font(20,i==0),fill=INK)
        return im

    def slate(lines):
        im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
        for i,line in enumerate(lines):d.text((110,250+i*110),line,font=font(44 if i==0 else 32,i==0),fill=INK)
        return im
    path=OUT/'furniture_collisions_and_forces.mp4';assert not path.exists()
    proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}',
        '-r',str(FPS),'-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    start=slate(['Can the robot physically touch this furniture?','14 layouts: seats, arms, backs, stools and tables.','First:3 controlled presses and an empty-space control per layout.','Then:full20-second robot attempts on layouts that passed.','4090 only; current PPO completed before these tests.'])
    for _ in range(30):proc.stdin.write(start.tobytes())
    for phase,count in [('probe',240),('robot',200)]:
        if phase=='robot':
            mid=slate(['Collision tests and robot success are different.','A solid, correctly measured surface can still defeat the policy.','The next segment shows the complete robot attempts.','Missing or conflicting contact normals never receive a PASS.'])
            for _ in range(30):proc.stdin.write(mid.tobytes())
        for tick in range(count):
            frame=grid(phase,tick);proc.stdin.write(frame.tobytes())
            if tick in [35,count-1]:frame.save(OUT/f'{phase}_{tick:03d}.png')
    clean=sum(t['audit']['clean'] is True for t in trials.values())
    end=slate(['Measured results',f"Collision-validated layouts: {len(result['validated_layouts'])}/14",
        f"Robot clean successes: {clean}/{len(trials)} tested; skipped layouts are not successes.",
        'Video replays all24s of probes and all20s of robot trials at1x.',
        'Original72-case benchmark and fresh24-case holdout unchanged.'])
    for _ in range(30):proc.stdin.write(end.tobytes())
    proc.stdin.close();assert proc.wait()==0
    info=__import__('json').loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=width,height,nb_frames','-of','json',str(path)]))
    assert float(info['format']['duration'])==53 and int(info['streams'][0]['nb_frames'])==530
    write(OUT/'video_manifest.json',dict(path=str(path),sha256=sha(path),duration_s=53,fps=10,frames=530,
        probe_seconds_shown=24,robot_seconds_shown=20,playback_speed=1,extra_simulation_steps=0,
        contact_measurement_dt_s=.02,display_dt_s=.1,collision_validated=len(result['validated_layouts']),
        robot_trials=len(trials),robot_clean=clean,video_probe=info,
        geometry_scope='Exact authored triangle furniture mesh; runtime USD robot collider/hull display, not bit-exact PhysX cooking certification.'))

if __name__=='__main__':main()
