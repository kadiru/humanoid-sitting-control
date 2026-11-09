"""CPU video of fresh recorded trajectories. Never fabricates contacts or motion."""
import functools,json,subprocess
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation
from paths import ROOT,OUT,ASSETS,BODY,HERE,read,write,sha
W,H,FPS=2560,1440,10;CW,CH=628,310
BG='#edf2f5';INK='#233b4d';RED='#bf403e';GREEN='#167657'
@functools.lru_cache(None)
def font(n,bold=False):return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans'+('-Bold' if bold else '')+'.ttf',n)
def main():
    cases=read(ASSETS/'cases.json');media=OUT/'media';media.mkdir(exist_ok=False)
    bodies=read(HERE/'render_colliders.json')['shapes'];records={};cache={};audits={};depths={}
    have_body=(OUT/'body_support/independent_audit.json').exists()
    body=read(OUT/'body_support/measurements.json') if have_body else []
    br=read(OUT/'body_support/result.json') if have_body else {'checks':[],'cases':[]}
    bm=read(OUT/'body_support/manifest.json') if have_body else None
    body_index={c['id']:i for i,c in enumerate(br['cases'])}
    view=np.array([[.8,.6,0.],[-.18,.24,.954]]);depth_axis=np.cross(view[0],view[1])
    for c in cases:
        folder=OUT/('robot_'+c['id']);audits[c['id']]=read(folder/'independent_audit.json')
        with np.load(folder/'trajectory.npz') as z:records[c['id']]={k:z[k][4::5,0] for k in ['body','force','alive','slip_ticks'] if k in z}
        records[c['id']]['body_names']=read(folder/'results.json')['body_names']
        records[c['id']]['contact_bodies']=read(folder/'manifest.json')['contact_bodies']
        with np.load(folder/'frame_0.npz') as z:depths[c['id']]={k:z[k] for k in ['rgb','depth','selected_pixels']}
        z=np.load(c['mesh_npz']);v=z['vertices'];tri=v[z['faces']]
        uv=np.vstack([v,[[.5,-.35,0],[.5,.35,1.35]]])@view.T
        sc=min((CW-42)/np.ptp(uv[:,0]),185/np.ptp(uv[:,1]));of=np.array([CW/2-sc*(uv[:,0].max()+uv[:,0].min())/2,255+sc*uv[:,1].min()])
        def proj(v,sc=sc,of=of):
            q=np.asarray(v)@view.T;return np.stack([of[0]+sc*q[...,0],of[1]-sc*q[...,1]],-1)
        im=Image.new('RGB',(CW,CH),'white');d=ImageDraw.Draw(im)
        for t in tri[np.argsort(tri.mean(1)@depth_axis)]:d.polygon([tuple(x) for x in proj(t)],fill='#c6b99d',outline='#a59d91')
        cache[c['id']]=(im,proj)
    def arrow(draw,proj,p,force):
        p=np.asarray(p);f=np.asarray(force)
        if not np.isfinite(p).all() or np.linalg.norm(f)<.1:return
        start,end=proj(np.array([p,p+f*.002]));draw.line([tuple(start),tuple(end)],fill=RED,width=3)
        delta=end-start;length=np.linalg.norm(delta)
        if length>4:
            u=delta/length;v=np.array([-u[1],u[0]]);draw.polygon([tuple(end),tuple(end-8*u+4*v),tuple(end-8*u-4*v)],fill=RED)
    def draw_shapes(draw,proj,shapes,pose):
        R=Rotation.from_quat(pose[3:]).as_matrix();ts=[]
        for s in shapes:
            v=np.asarray(s['vertices'])@R.T+pose[:3]
            for t in v[np.asarray(s['faces'])]:ts.append((float(t.mean(0)@depth_axis),t))
        for _,t in sorted(ts,key=lambda a:a[0]):draw.polygon([tuple(x) for x in proj(t)],fill='#92b8cc',outline='#46748e')
    def grid(kind,tick):
        im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
        title={'robot':'Full20s robot attempts: unchanged physical success criteria','body':'Controlled body-shape loads: separate from autonomous success','depth':'Fresh rear-camera images: green pixels are the observed selected patch'}[kind]
        d.text((22,14),title,font=font(30,True),fill=INK)
        d.text((22,55),'All14 layouts. Rigid fixed furniture; collision meshes shown. Red arrows: measured contact force,2mm/N.',font=font(22),fill=INK)
        d.text((22,88),'Saved physics replay at1x. A failed or unvalidated attempt stays failed or unvalidated.',font=font(20),fill=INK)
        for ci,c in enumerate(cases):
            panel,proj=cache[c['id']];panel=panel.copy();q=ImageDraw.Draw(panel)
            q.text((12,8),(c['title']+' / '+c['approach'].replace('_',' '))[:48],font=font(18,True),fill=INK)
            if kind=='depth':
                cap=depths[c['id']];rgb=cap['rgb'][...,:3].astype('uint8').copy();pix=cap['selected_pixels']
                if len(pix):rgb[tuple(pix.T)]=(rgb[tuple(pix.T)].astype(float)*.35+np.array([20,220,75])*.65).astype('uint8')
                picture=Image.fromarray(rgb).resize((304,228),Image.Resampling.NEAREST)
                dd=cap['depth'];z=np.clip((dd-.08)/1.5,0,1);z[~np.isfinite(z)]=1
                gray=np.stack((255*(1-z),190*(1-z),255*z),-1).astype('uint8')
                panel.paste(picture,(8,52));panel.paste(Image.fromarray(gray).resize((304,228),Image.Resampling.NEAREST),(316,52));q=ImageDraw.Draw(panel)
                r=read(OUT/('robot_'+c['id'])/'depth_audit.json')['frames'][0]
                fraction=r['selected_target_fraction'];q.text((12,32),f"mesh match {100*fraction:.1f}%" if fraction is not None else 'NO OBSERVED PATCH',font=font(17,True),fill=INK)
            elif kind=='robot':
                data=records[c['id']];poses=data['body'][tick]
                for bi,b in enumerate(data['body_names']):
                    shapes=[s for s in bodies if s['body']==b]
                    if shapes:draw_shapes(q,proj,shapes,poses[bi])
                # Aggregate force acts at body origins for legibility; force is
                # measured, but arrow position is explicitly a display anchor.
                names=data['contact_bodies']
                for j,b in enumerate(names):
                    if b in data['body_names']:arrow(q,proj,poses[data['body_names'].index(b),:3],data['force'][tick,j])
                a=audits[c['id']];label='CLEAN PASS' if a['clean'] is True else 'UNVALIDATED CONTACT SCORE' if a['clean'] is None else 'FAIL'
                q.text((12,34),label,font=font(17,True),fill=GREEN if a['clean'] else RED)
                q.text((12,278),f"t={(tick*5+5)*.02:5.2f}s | hold {a['final_hold_s']:.2f}s | slip ticks {a['slip_ticks']}",font=font(16),fill=INK)
            elif c['id'] in body_index:
                j,k=divmod(tick,60);i=body_index[c['id']];row=body[j][k*5+4][i];pose=np.asarray(row['pose']);pose[:3]-=[0,i*8,0]
                draw_shapes(q,proj,bm['bodies'][row['body']],pose)
                cp=np.mean(row['points'],axis=0) if row['points'] else pose[:3];arrow(q,proj,cp,row['force'])
                result=next(r for r in br['checks'] if r['case']==c['id'] and r['round']==j)
                q.text((12,34),'SUPPORT PASS' if result['physical_support'] else 'SUPPORT FAIL',font=font(17,True),fill=GREEN if result['physical_support'] else RED)
                q.text((12,58),'Intended patch: '+('PASS' if result['intended_contact_patch'] else 'EDGE / OTHER FACE'),font=font(16,True),fill=GREEN if result['intended_contact_patch'] else RED)
                q.text((12,278),f"t={row['t']:4.2f}s | {row['body']} | {np.linalg.norm(row['force']):.1f}N",font=font(15),fill=INK)
            else:q.text((12,45),'NO OBSERVED INTERIOR: FIXTURE SKIPPED',font=font(17,True),fill=RED)
            q.rectangle((1,1,CW-2,CH-2),outline='#becbd3',width=2);im.paste(panel,(14+(CW+8)*(ci%4),135+(CH+8)*(ci//4)))
        x=14+2*(CW+8);y=135+3*(CH+8)
        lines=['READ THE LABELS','Depth recognition is not sitting success.','Robot:20s,5s stable hold required.','No loaded slip; legal contacts only.','Body fixtures:gravity off,pose held.','Force arrows on robot use body origins.','4070 remains reserved for AIWorker.']
        for i,line in enumerate(lines):d.text((x+12,y+12+i*30),line,font=font(20,i==0),fill=INK)
        return im
    def slate(lines):
        im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
        for i,line in enumerate(lines):d.text((110,250+i*100),line,font=font(38 if i==0 else 28,i==0),fill=INK)
        return im
    path=media/'updated_furniture_depth_contacts.mp4'
    proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(path)],stdin=subprocess.PIPE)
    frames=0
    def emit(frame,count=1):
        nonlocal frames
        for _ in range(count):proc.stdin.write(frame.tobytes());frames+=1
    emit(slate(['Fresh furniture retest after PPO','Corrected depth alignment and task-directed surface recognition.','14 layouts:seats,arms,backs,stools andtables.','Actual saved simulation; unchanged physical/action thresholds.']),30)
    frame=grid('depth',0);frame.save(media/'depth_recognition.jpg');emit(frame,60)
    for tick in range(200):
        frame=grid('robot',tick);emit(frame)
        if tick in [0,49,99,199]:frame.save(media/f'robot_{tick:03d}.png')
    if have_body:
        emit(slate(['Controlled load tests','These body-shaped fixtures isolate contact and load capacity.','They do not demonstrate free-standing robot balance.']),20)
        for tick in range(180):
            frame=grid('body',tick);emit(frame)
            if tick in [35,95,179]:frame.save(media/f'body_{tick:03d}.png')
    clean=sum(a['clean'] is True for a in audits.values());invalid=sum(a['clean'] is None for a in audits.values())
    emit(slate(['Measured outcomes',f'Autonomous furniture clean:{clean}/14; unvalidated contact scores:{invalid}.',f"Controlled load passes:{sum(r['physical_support'] for r in br['checks'])}/{len(br['checks'])}.",'Correct recognition does not by itself repair the policy.']),30)
    proc.stdin.close();assert proc.wait()==0
    info=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=width,height,nb_frames','-of','json',str(path)]))
    assert float(info['format']['duration'])==frames/FPS<60 and int(info['streams'][0]['nb_frames'])==frames
    write(media/'video_manifest.json',dict(path=str(path),sha256=sha(path),duration_s=frames/FPS,frames=frames,
        robot_seconds=20,body_seconds=18 if have_body else 0,playback_speed=1,extra_simulation_steps=0,
        force_display='2mm/N; robot forces anchored at body origins; fixture forces at measured contact mean',
        audit_bindings={c['id']:sha(OUT/('robot_'+c['id'])/'independent_audit.json') for c in cases}))
if __name__=='__main__':main()
