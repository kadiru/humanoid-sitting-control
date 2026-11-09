"""Offline replay of saved poses and independently counted full-trial scores."""
from pathlib import Path
import json,csv,hashlib,subprocess,functools,time
import numpy as np
from PIL import Image,ImageDraw,ImageFont

O=Path(__file__).resolve().parent
D=O/'inputs';M=D/'contact_reward_comparison_128_v4'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
font=functools.lru_cache(None)(lambda s,b=False:ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf' if b else '/System/Library/Fonts/Supplemental/Arial.ttf',s))
def rot(q):
    x,y,z,w=np.asarray(q,dtype=float)/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
assert np.array_equal(rot([0,0,0,1]),np.eye(3))
def clean(r):return bool(r['success'] and not any(r['failures'].values()) and not r['slip_ticks'] and not r['illegal_ticks'])
items={};scores=[];bindings={}
for bank in ['known_old','known_recent','known_transfer']:
    p=M/f'eval_revised_63111_064_{bank}'
    assert read(p/'independent_audit.json')['passed']
    r=read(p/'results.json');tr=dict(np.load(p/'trace.npz'));tk=dict(np.load(p/'every_tick.npz'));ra=read(p/'reward_audit.json')[0]
    br=read(D/f'current_retention_09/{bank}/candidate/results.json');base={x['case']:x for x in br['episodes']}
    active=np.linalg.norm(tk['force'],axis=-1)>.01*tk['weight'][...,None]
    speeds=np.where(active,tk['slip'],0).max(-1)
    assert np.array_equal((speeds<=.02),tk['no_slip'])
    for f in p.iterdir():bindings[str(f.relative_to(O))]=sha(f)
    for i,row in enumerate(r['episodes']):
        case=row['case'];key=f'{bank}/{case}'
        slip=(~tk['no_slip'][:,case]);illegal=~tk['legal'][:,case]
        assert int(slip.sum())==row['slip_ticks'] and int(illegal.sum())==row['illegal_ticks']
        assert bool(tk['failure'][:,case].any())==any(row['failures'].values())
        held=0
        # The label adapter recomputes instantaneous good with alive=True.
        # The actual episode hold also requires the latched alive state.
        for good in (tk['good'][:,case] & tk['alive'][:,case]):held=held+1 if good else 0
        assert abs(held*.02-row['final_hold_s'])<1e-6
        assert row['success']==(held>=250 and not any(row['failures'].values()) and row['steps']==1000)
        components=sum(v[i] for v in ra['component_totals_raw'].values())+sum(v[i] for v in ra['event_totals_raw'].values())+ra['terminal_raw'][i]
        assert abs(components/50-ra['full_episode_return'][i])<2e-7
        bad=np.flatnonzero(slip);contact=np.flatnonzero(tk['wrong'][:,case]>10)
        reasons=[]
        if len(bad):reasons.append('slip')
        reasons.extend(k for k,v in row['failures'].items() if v)
        if row['illegal_ticks']:reasons.append('illegal patch')
        if row['final_hold_s']<5:reasons.append('insufficient final hold')
        geom=read(p/'geometry.json')[row['geometry_index']]
        score=dict(case=key,baseline_score=int(clean(base[case])),strict_score=int(clean(row)),stable_hold_pass=bool(row['success']),final_hold_s=row['final_hold_s'],slip_ticks=row['slip_ticks'],slip_duration_s=row['slip_ticks']*.02,peak_loaded_slip_m_s=float(speeds[:,case].max()),first_slip_s=float(tk['t'][bad[0]]) if len(bad) else None,forbidden_contact=bool(row['failures']['forbidden']),peak_wrong_force_N=float(tk['wrong'][:,case].max()),first_wrong_contact_s=float(tk['t'][contact[0]]) if len(contact) else None,illegal_ticks=row['illegal_ticks'],failure_flags=json.dumps(row['failures'],sort_keys=True),reasons='; '.join(reasons) or 'clean',revised_training_return=float(ra['full_episode_return'][i]),height_cm=100*geom['height'],slope_deg=geom['pitch_deg'],classification='protected success' if clean(row) else ('regression' if clean(base[case]) else 'original failure'))
        scores.append(score);items[key]=dict(row=row,score=score,results=r,trace=tr,ticks=tk,speed=speeds[:,case],geometry=geom,inertial=read(p/'static_inertial_model.json'),case=case)
assert len(scores)==72 and sum(x['strict_score'] for x in scores)==59
original=[x['case'] for x in scores if x['classification']=='original failure'];regress=[x['case'] for x in scores if x['classification']=='regression']
assert len(original)==8 and len(regress)==5
with (O/'scores_all72.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(scores[0]));w.writeheader();w.writerows(scores)
(O/'scores_all72.json').write_text(json.dumps(scores,indent=2)+'\n')
W,H=1920,1080;CW,CH=461,464;FPS=10
INK='#213547';RED='#ad3340';GREEN='#13785b';BG='#eef3f6'
def card(key,ti):
    it=items[key];s=it['score'];row=it['row'];ci=it['case'];g=it['geometry'];tr=it['trace'];tk=it['ticks'];t=float(tr['t'][ti])
    im=Image.new('RGB',(CW,CH),'white');d=ImageDraw.Draw(im);color=GREEN if s['strict_score'] else RED
    d.rounded_rectangle((2,2,CW-3,CH-3),12,outline=color,width=4)
    d.text((12,10),key.replace('known_','')+' | '+s['classification'],font=font(19,True),fill=INK)
    d.text((12,38),f"{'PASS' if s['strict_score'] else 'FAIL'} {s['strict_score']}/1   Reward {s['revised_training_return']:+.3f}",font=font(21,True),fill=color)
    d.text((12,68),f"Slip {s['slip_duration_s']:.2f}s | peak {s['peak_loaded_slip_m_s']*100:.3f} cm/s",font=font(16),fill=INK)
    d.text((12,91),f"Final hold {s['final_hold_s']:.2f}s | forbidden contact {'YES' if s['forbidden_contact'] else 'no'}",font=font(16),fill=INK)
    names=it['results']['body_names'];xyz=tr['body'][ti,ci,:,:3]-np.asarray(it['results']['origins'][ci]);scale=145;cx=CW*.61;floor=332
    def proj(p):
        p=np.asarray(p);return np.stack([cx+scale*(.90*p[...,0]+.44*p[...,1]),floor-scale*(p[...,2]+.12*(-.44*p[...,0]+.90*p[...,1]))],-1)
    def line(p,col,width=2):d.line([tuple(a) for a in proj(p)],fill=col,width=width)
    for x in np.arange(-1.,.71,.25):line([[x,-.65,0],[x,.65,0]],'#e0e7ec')
    corners=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]])*np.array([g['depth']/2,g['width']/2,1]);corners=corners@rot(g['rotation_xyzw']).T+np.asarray(g['top_center'])
    d.polygon([tuple(x) for x in proj(corners)],fill='#d4b778',outline='#9c7c39')
    for corner in corners:line([corner,[corner[0],corner[1],0]],'#b09259',3)
    torso_ids=[names.index(n) for n in ['left_hip_roll_link','left_shoulder_pitch_link','right_shoulder_pitch_link','right_hip_roll_link']]
    d.polygon([tuple(x) for x in proj(xyz[torso_ids])],fill='#bed0db',outline='#426078')
    for side,col in [('left','#237c94'),('right','#526aa1')]:
        for chain in [['pelvis']+[side+k for k in ['_hip_pitch_link','_hip_roll_link','_hip_yaw_link','_knee_link','_ankle_pitch_link','_ankle_roll_link']],['torso_link']+[side+k for k in ['_shoulder_pitch_link','_shoulder_roll_link','_shoulder_yaw_link','_elbow_pitch_link','_elbow_roll_link','_palm_link']]]:
            ids=[names.index(n) for n in chain if n in names];line(xyz[ids],col,7)
            for x,y in proj(xyz[ids]):d.ellipse((x-3,y-3,x+3,y+3),fill=col)
    hi=names.index('head_link');head=xyz[hi]+rot(tr['body'][ti,ci,hi,3:7])@np.asarray(it['inertial']['local_com'][hi]);shoulders=xyz[[names.index(s+'_shoulder_pitch_link') for s in ['left','right']]].mean(0)
    line([shoulders,head],'#426078',8);x,y=proj(head);d.ellipse((x-10,y-10,x+10,y+10),fill='#426078')
    d.text((12,120),f"Seat {s['height_cm']:g}cm / {s['slope_deg']:+g}deg",font=font(14),fill='#566c7c')
    # Every20ms measurement is plotted; pose replay uses exact100ms saved frames.
    xx=12+tk['t']/20*(CW-24);v=it['speed'];ymax=max(.035,float(v.max())*1.15);yy=426-v/ymax*45
    threshold=426-.02/ymax*45;d.line((12,threshold,CW-12,threshold),fill='#cb8c24',width=2)
    d.line(list(zip(xx,yy)),fill='#69808f',width=2);d.line((12+t/20*(CW-24),378,12+t/20*(CW-24),430),fill='#1e495c',width=2)
    for j in np.flatnonzero(v>.02):d.line((xx[j],yy[j],xx[j],426),fill=RED,width=2)
    d.text((12,433),'Loaded slip speed: amber line = 2 cm/s limit',font=font(14),fill='#566c7c')
    if s['first_slip_s'] is not None and t+1e-6>=s['first_slip_s']:
        text=f"Slip began {s['first_slip_s']:.2f}s"
        if s['first_wrong_contact_s'] is not None and t>=s['first_wrong_contact_s']:text+=' | housing contact'
        d.rounded_rectangle((10,345,CW-10,374),5,fill='#fbe7e9');d.text((16,349),text,font=font(16,True),fill=RED)
    return im
def info(lines):
    im=Image.new('RGB',(CW,CH),'#e5edf2');d=ImageDraw.Draw(im)
    for i,line in enumerate(lines):d.text((18,22+i*35),line,font=font(18,i==0),fill=INK)
    return im
pages=[original,regress+['known_old/0']]
def frame(page,ti):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    d.text((20,14),'REVISED PPO | seed63111 | 64 updates | all13 failed cases',font=font(31,True),fill=INK)
    d.text((20,57),f"{'Original8 failures' if page==0 else 'Five regressions + one clean reference'} | recorded t={(ti+1)*.1:4.1f}/20s | 1x speed",font=font(22),fill=INK)
    d.text((20,91),'Saved simulation poses, schematic replay; contact measurements sampled at20ms. Full-trial scores stay visible.',font=font(19),fill='#566c7c')
    for i,k in enumerate(pages[page]):im.paste(card(k,ti),(20+i%4*473,132+i//4*476))
    if page==1:
        texts=[['HOW A TRIAL PASSES','Complete20 seconds.','Final5 seconds supported,','quiet and within limits.','No slip at any loaded contact.','No forbidden contact, fall,','hard impact or limit violation.','','Score: PASS1 / FAIL0.','Reward is a different number.'],['READ THE VIDEO','A20ms slip can fail a trial','even when sitting looks stable.','Red graph spikes show slips.','A slip can occur between','the100ms pose samples.','','These are recorded poses,','not camera footage or a','new simulation.']]
        for i,lines in enumerate(texts,6):im.paste(info(lines),(20+i%4*473,132+i//4*476))
    return im
def title(lines):
    im=Image.new('RGB',(W,H),BG);d=ImageDraw.Draw(im)
    for i,txt in enumerate(lines):d.text((100,180+i*95),txt,font=font(42 if i==0 else 29,i==0),fill=INK)
    return im
start=title(['Why these sitting attempts fail','Revised rewards / seed63111 / checkpoint64','59/72 clean |70/72 stable |13 slipping |2 forbidden contacts','Page1: all8 original failures. Page2: all5 regressions + clean reference.','Each page replays the entire20-second attempt at normal speed.','Schematic from saved simulation poses; no training interrupted.'])
end=title(['A failed score does not always mean a fall','All13 failures shown. Two include forbidden housing contact.','The other11 finish stable but slip briefly during the attempt.','Current strict score:59/72 =81.9%. Baseline:64/72 =88.9%.','No threshold changed; no repaired candidate claimed.','Exact per-case scores and reward components accompany this video.'])
video=O/'failing_cases_revised64_v2.mp4'
assert not video.exists()
proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(FPS),'-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE,stderr=(O/'ffmpeg.log').open('w'))
for _ in range(30):proc.stdin.write(start.tobytes())
for page in [0,1]:
    for ti in range(200):
        im=frame(page,ti);proc.stdin.write(im.tobytes())
        if ti in [103,123,199]:im.save(O/f'page{page}_t{(ti+1)*.1:.1f}.png')
    print('page',page+1,'complete',flush=True)
for _ in range(30):proc.stdin.write(end.tobytes())
proc.stdin.close();assert proc.wait()==0
probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=width,height,nb_frames,codec_name','-of','json',str(video)]))
assert float(probe['format']['duration'])==46 and probe['streams'][0]['nb_frames']=='460'
manifest={'scope':'All13 failed revised63111 checkpoint64 full development episodes and one clean reference; not all historical attempts.','pose_samples_per_episode':200,'pose_dt_s':.1,'measurement_samples_per_episode':1000,'measurement_dt_s':.02,'pages':pages,'playback_speed':1,'duration_s':46,'strict_clean':59,'stable':70,'failed':13,'original_failures':8,'regressions':5,'scoring_checks':'All72 slip and illegal counts, physical flags, final hold, stable success and reward-component totals checked against saved arrays.','source_bindings':bindings,'renderer_sha256':sha(Path(__file__)),'video_sha256':sha(video),'probe':probe,'new_simulation_steps':0,'optimizer_steps':0,'training_interrupted':False}
(O/'video_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({'video':str(video),'duration_s':46,'failed_cases':13,'scores':str(O/'scores_all72.csv')}))
