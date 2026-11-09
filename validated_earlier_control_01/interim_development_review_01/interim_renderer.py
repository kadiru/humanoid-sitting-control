"""Under-one-minute, complete final-episode coverage from saved simulator states.

Offline 3D schematic; no synthesized outcomes and no rerun/selection of trials.
"""
import argparse,json,math,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial.transform import Rotation

FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
BOLD='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
def font(size,bold=False):return ImageFont.truetype(BOLD if bold else FONT,size)
def read(p):return json.loads(Path(p).read_text())
def label(row):
    meanings={'fall':'fall','forbidden':'unwanted body contact','torque':'torque limit','impact':'hard impact','joint_limit':'joint limit'}
    reasons=[meanings.get(k,k) for k,value in row['failures'].items() if value]
    if row.get('slip_ticks',0):reasons.append('slipping')
    if row.get('illegal_ticks',0):reasons.append('unwanted contact')
    if not row['success'] and not reasons:reasons.append('stable support not reached')
    return ('FAILURE' if reasons or not row['success'] else 'SUCCESS'),reasons
def name(key):
    if key=='reference':return 'Repaired reference'
    return key.replace('parent_','Current PPO ').replace('final_','Final PPO ').replace('selected_','Selected PPO ')
def sources(paths):
    data={}
    for bank,items in paths.items():
        data[bank]={}
        for key,folder in items.items():
            p=Path(folder);results=read(p/'results.json');trace=dict(np.load(p/'trace.npz'));ticks=dict(np.load(p/'every_tick.npz'))
            data[bank][key]=dict(path=str(p),results=results,trace=trace,ticks=ticks,inertial=read(p/'static_inertial_model.json'),geometry=read(p/'geometry.json'),rows={x['case']:x for x in results['episodes']})
    return data
def draw_card(item,case,time_index,key,w,h):
    row=item['rows'][case];outcome,reasons=label(row);border='#a32c35' if outcome=='FAILURE' else '#197553'
    card=Image.new('RGB',(w,h),'#ffffff');d=ImageDraw.Draw(card)
    d.rounded_rectangle((2,2,w-3,h-3),radius=10,outline=border,width=5)
    d.text((15,11),name(key),font=font(22,True),fill='#233a4b');d.text((15,42),outcome,font=font(25,True),fill=border)
    reason=', '.join(reasons).replace('_',' ') or 'Stable support; no slip/contact'
    if len(reason)>48:reason=reason[:45]+'...'
    d.text((15,75),reason,font=font(15),fill=border)
    margin=row.get('traction_bad_ticks',0)
    if margin:d.text((15,96),f'Margin warning: {margin} physics ticks',font=font(14),fill='#8b650c')
    trace=item['trace'];results=item['results'];names=results['body_names'];origins=np.array(results['origins'])
    xyz=trace['body'][time_index,case,:,:3]-origins[case];g=item['geometry'][row['geometry_index']]
    # Constant oblique projection preserves left/right separation and a common scale.
    scale=min((w-40)/1.65,(h-180)/1.55);cx=w*.53;floor=h-50
    def project(points):
        p=np.asarray(points);u=.90*p[...,0]+.44*p[...,1];v=p[...,2]+.12*(-.44*p[...,0]+.90*p[...,1])
        return np.stack([cx+scale*u,floor-scale*v],-1)
    def line(points,color,width=2):d.line([tuple(p) for p in project(points)],fill=color,width=width)
    for x in np.arange(-1.,.71,.2):line([[x,-.65,0],[x,.65,0]],'#e1e8eb')
    for y in np.arange(-.6,.61,.2):line([[-1.,y,0],[.7,y,0]],'#e1e8eb')
    rot=Rotation.from_quat(g['rotation_xyzw']).as_matrix();center=np.array(g['top_center'])
    corners=np.array([[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]])*np.array([g['depth']/2,g['width']/2,1])
    corners=corners@rot.T+center;d.polygon([tuple(p) for p in project(corners)],fill='#ddbd7c',outline='#997438')
    # Fixed-link origins coincide with the torso origin. Use the recorded
    # rotation and exported physical COM for the head, not that origin.
    head=names.index('head_link');q=trace['body'][time_index,case,head,3:7]
    headxyz=xyz[head]+Rotation.from_quat(q[[1,2,3,0]]).apply(item['inertial']['local_com'][head])
    torso_ids=[names.index(n) for n in ['left_hip_roll_link','left_shoulder_pitch_link','right_shoulder_pitch_link','right_hip_roll_link']]
    d.polygon([tuple(p) for p in project(xyz[torso_ids])],fill='#b0c3d0',outline='#314f65')
    shoulders=xyz[[names.index(s+'_shoulder_pitch_link') for s in ['left','right']]].mean(0)
    line([xyz[names.index('pelvis')],shoulders,headxyz],'#314f65',9)
    chains=[]
    for side in ['left','right']:
        chains.append(['pelvis']+[side+k for k in ['_hip_pitch_link','_hip_roll_link','_hip_yaw_link','_knee_link','_ankle_pitch_link','_ankle_roll_link']])
        chains.append(['torso_link']+[side+k for k in ['_shoulder_pitch_link','_shoulder_roll_link','_shoulder_yaw_link','_elbow_pitch_link','_elbow_roll_link','_palm_link']])
    colors=['#237c94','#4a6399','#237c94','#4a6399']
    for chain,color in zip(chains,colors):
        ids=[names.index(n) for n in chain if n in names]
        if len(ids)>1:line(xyz[ids],color,7 if w<600 else 10)
        for p in project(xyz[ids]):d.ellipse((p[0]-4,p[1]-4,p[0]+4,p[1]+4),fill=color)
    if 'head_link' in names:
        p=project(headxyz);radius=max(8,scale*.06);d.ellipse((p[0]-radius,p[1]-radius,p[0]+radius,p[1]+radius),fill='#314f65')
    t=float(trace['t'][time_index]);tick=item['ticks'];near=int(np.argmin(abs(tick['t']-t)))
    alerts=[]
    lo=max(0,near-3);hi=min(len(tick['t']),near+4)
    if not tick['no_slip'][lo:hi,case].all():alerts.append('SLIP EVENT')
    if (tick['wrong'][lo:hi,case]>10).any():alerts.append('CONTACT EVENT')
    if tick['failure'][lo:hi,case].any():alerts.append('PHYSICAL FAILURE')
    if alerts:d.rectangle((8,h-25,w-8,h-6),fill='#a32c35');d.text((14,h-25),' / '.join(alerts),font=font(13,True),fill='white')
    else:
        rms=float(np.sqrt(trace['effort'][time_index,case]/37));d.text((14,h-24),f'Instantaneous all-joint RMS {rms:.2f} Nm',font=font(13),fill='#455b68')
    return card

def render(paths,output,smoke=False):
    output=Path(output);assert not output.exists();output.mkdir(parents=True)
    data=sources(paths);keys=list(next(iter(data.values())));validation=all(k.startswith('archived') for k in data)
    assert all(list(x)==keys for x in data.values())
    pages=[];episodes=[]
    for bank,items in data.items():
        cases=sorted(next(iter(items.values()))['rows'])
        assert all(sorted(x['rows'])==cases for x in items.values())
        for case in cases:
            critical=set([0,199])
            for key,item in items.items():
                row=item['rows'][case];outcome,reasons=label(row);tick=item['ticks'];times=[]
                for mask in [~tick['no_slip'][:,case],tick['wrong'][:,case]>10,tick['failure'][:,case]]:
                    bad=np.flatnonzero(mask)
                    if len(bad):times.append(float(tick['t'][bad[0]]))
                critical.update(int(np.argmin(abs(item['trace']['t']-t))) for t in times)
                episodes.append(dict(bank=bank,controller=key,case=case,label=outcome,reasons=reasons,stable_success=row['success'],slip_ticks=row.get('slip_ticks',0),illegal_ticks=row.get('illegal_ticks',0),margin_ticks=row.get('traction_bad_ticks',0),first_events_s=times,source=item['path']))
            pages.append(dict(bank=bank,case=case,critical=sorted(critical)))
    if smoke:
        failing={(x['bank'],x['case']) for x in episodes if x['label']=='FAILURE'}
        pages=[next((p for p in pages if (p['bank'],p['case']) in failing),pages[0])]
        episodes=[x for x in episodes if x['bank']==pages[0]['bank'] and x['case']==pages[0]['case']]
    width,height,fps=1920,1080,30;intro=60 if not smoke else 0;outro=60 if not smoke else 0
    per_case=max(1,int((59*fps-intro-outro)/len(pages)));per_case=min(per_case,90) if smoke else per_case
    total_frames=intro+outro+per_case*len(pages);assert total_frames/fps<60
    speed=20/(per_case/fps);cols=min(4,math.ceil(len(keys)/2)) if len(keys)>4 else len(keys);rows=math.ceil(len(keys)/cols)
    cardw=(width-40-(cols-1)*12)//cols;cardh=(height-155-(rows-1)*12)//rows
    args=['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{width}x{height}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(output/'summary.mp4')]
    log=(output/'ffmpeg.log').open('w');proc=subprocess.Popen(args,stdin=subprocess.PIPE,stderr=log)
    def send(im):proc.stdin.write(im.tobytes())
    def title_card(lines):
        im=Image.new('RGB',(width,height),'#edf3f5');d=ImageDraw.Draw(im)
        for i,text in enumerate(lines):d.text((90,150+85*i),text,font=font(38 if i==0 else 28,i==0),fill='#213a49')
        return im
    startcard=title_card(['INTERIM: six-update development comparison','Green SUCCESS: stable support, no slipping or unwanted contacts.','Red FAILURE: physical failure, slip/contact, or missing stable support.','Amber margin warnings are reported separately.',f'{len(episodes)} episodes / {len(pages)} conditions; synchronized {speed:.1f}x fast-forward.','Offline 3D schematic from recorded simulator states.'])
    if validation:startcard=title_card(['ARCHIVED DATA — VIDEO VALIDATION','Recorded successes and failures from completed studies.','This tests the renderer and episode-coverage audit.',f'{len(episodes)} archived episodes; {speed:.1f}x fast-forward.','Not a result of the new earlier-control PPO study.'])
    for _ in range(intro):send(startcard)
    page_manifest=[]
    for pi,page in enumerate(pages):
        needed=set(page['critical']);assert len(needed)<=per_case,(page,'too many distinct first-event frames')
        for index in np.linspace(0,199,per_case*3).round().astype(int):
            if len(needed)==per_case:break
            needed.add(int(index))
        # If critical points displaced uniform points, fill the complete horizon evenly.
        uniform=np.linspace(0,199,per_case).round().astype(int).tolist();indices=sorted(set(page['critical']+uniform))
        while len(indices)>per_case:
            candidates=[i for i,x in enumerate(indices) if x not in page['critical']]
            index=min(candidates,key=lambda i:min(indices[i]-indices[i-1] if i else 999,indices[i+1]-indices[i] if i+1<len(indices) else 999));indices.pop(index)
        while len(indices)<per_case:
            gaps=[(b-a,i) for i,(a,b) in enumerate(zip(indices,indices[1:]))];gap,index=max(gaps)
            if gap<=1:indices.append(indices[-1])
            else:indices.insert(index+1,(indices[index]+indices[index+1])//2)
        first=next(iter(data[page['bank']].values()));row=first['rows'][page['case']];g=first['geometry'][row['geometry_index']]
        for ti in indices:
            im=Image.new('RGB',(width,height),'#edf3f5');d=ImageDraw.Draw(im)
            d.text((22,12),f"INTERIM / 6 UPDATES | {page['bank'].upper()} | case {page['case']:02d} | seat {g['height']*100:.1f} cm | slope {g['pitch_deg']:+.1f} deg",font=font(29,True),fill='#213a49')
            d.text((22,55),f"Simulated {first['trace']['t'][ti]:4.1f} / 20.0 s | {speed:.1f}x | condition {pi+1}/{len(pages)} | recorded-state schematic",font=font(21),fill='#435c6a')
            for ki,key in enumerate(keys):
                card=draw_card(data[page['bank']][key],page['case'],ti,key,cardw,cardh);im.paste(card,(20+(ki%cols)*(cardw+12),100+(ki//cols)*(cardh+12)))
            d.text((22,height-34),'Labels describe the FULL trial. Failed episodes are retained. Torque is an effort proxy, not energy.',font=font(18),fill='#435c6a')
            if smoke or validation:d.text((width-620,55),'RENDERER VALIDATION / PRIOR STUDY',font=font(20,True),fill='#a32c35')
            send(im)
        page_manifest.append(dict(bank=page['bank'],case=page['case'],first_frame=intro+pi*per_case,frames=per_case,trace_indices=indices,required_event_indices=page['critical']))
    successes=sum(x['label']=='SUCCESS' for x in episodes);endcard=title_card(['All 90 interim development episodes included',f'{successes} SUCCESS labels | {len(episodes)-successes} FAILURE labels',f'{len(episodes)} episodes, {total_frames/fps:.2f} seconds.', 'Training continues. Wider unseen-surface tests have not run.','Reference-assisted control; two seeds and finite rigid surfaces.'])
    for _ in range(outro):send(endcard)
    proc.stdin.close();assert proc.wait()==0;log.close()
    probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,width,height,r_frame_rate,nb_frames','-of','json',str(output/'summary.mp4')]))
    assert float(probe['format']['duration'])<60 and probe['streams'][0]['codec_name']=='h264'
    manifest=dict(smoke_only=smoke,validation_only=validation,scope='Archived renderer validation' if validation else 'Interim block-0 development recordings; training exploration and unseen surfaces excluded',fps=fps,duration_s=total_frames/fps,fast_forward=speed,episodes=episodes,pages=page_manifest,source_paths=paths,probe=probe)
    (output/'coverage.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(dict(video=str(output/'summary.mp4'),seconds=total_frames/fps,episodes=len(episodes),failures=len(episodes)-successes)))
    return manifest
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verification',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');a=p.parse_args();render(read(a.verification)['paths'],a.output,a.smoke)
