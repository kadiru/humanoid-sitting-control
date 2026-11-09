"""All low-seat physical attempts to date, recorded-state schematic under60s."""
from pathlib import Path
import json,hashlib,subprocess,datetime,functools
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import render_positive_slope_video_v3 as base
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'all_attempts_video_250'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
font=functools.lru_cache(None)(base.font);base.font=font
ALIASES={'damped_original_01':'01 damping','clearance_damped_01':'01 deeper+damping','load_feedback_original_02':'02 load feedback','load_feedback_clearance_02':'02 deeper+load','normal_descent_03':'03 normal descent','hover_rotate_normal_03':'03 hover/rotate','normal_ankle_balance_04':'04 ankle balance','normal_load_balance_04':'04 load balance','normal_task_mild_05':'05 coordinated mild','normal_task_firm_05':'05 coordinated firm','normal_landing2s_06':'06 normal 2s','normal_landing3s_06':'06 normal 3s','normal_slow_orientation_08':'08 slow rotation','normal_counterbalance_13':'13 arm balance','joint_path_17':'17 joint path','tracking_headroom_31':'31 tracking rate','staged_approach_36':'36 staged','dual_view_approach_40':'40 dual view'}
ALIASES['fixed_foot_probe_134']='134 gated fixed-foot'
ALIASES['balanced_force_probe_150']='150 balanced forces'
ALIASES['balance_feedback_probe_164']='164 measured balance'
ALIASES['touchdown_probe_178']='178 measured settling'
ALIASES['bias_probe_210']='210 bounded bias'
ALIASES['centered_probe_234']='234 centered stance'
def main():
 O.mkdir(exist_ok=False);paths=[]
 for tr in sorted(D.glob('**/trace.npz')):
  p=tr.parent
  if (p/'results.json').is_file() and (p/'every_tick.npz').is_file():paths.append(p)
 # Full motion attempts first, then recorded diagnostic prefixes; all inventory retained.
 paths.sort(key=lambda p:(float(np.load(p/'trace.npz')['t'][-1])<19.99,str(p)))
 items=[];hashes={};episodes=[]
 for p in paths:
  r=read(p/'results.json');trace=dict(np.load(p/'trace.npz'));tick=dict(np.load(p/'every_tick.npz'));rows={x['case']:x for x in r['episodes']}
  if not rows:
   assert float(trace['t'][-1])<20 and trace['body'].shape[1]==24
   bp=D/'normal_landing3s_06/candidate';assert read(p/'geometry.json')==read(bp/'geometry.json')
   assert read(p/'manifest.json')['args']['bank']==str(D/'banks/probe')
   assert len(r['origins'])==24
   rows={x['case']:dict(case=x['case'],geometry_index=x['geometry_index'],success=False,failures={},slip_ticks=int((~tick['no_slip'][:,x['case']]).sum()),illegal_ticks=int((~tick['legal'][:,x['case']]).sum()),traction_bad_ticks=0) for x in read(bp/'results.json')['episodes']}
  assert sorted(rows)==list(range(24));rel=str(p.relative_to(D));parts=p.relative_to(D).parts
  if parts[0].startswith('current_retention_'):title='Ret'+parts[0][-2:]+' '+parts[1].replace('known_','')
  elif parts[0]=='sensor_prefix_v4_24':title='24 prefix '+parts[-1]
  elif parts[0]=='dual_view_diagnostic_39':title='39 camera prefix'
  elif parts[0]=='substep_diagnostic_189':title='189 substep prefix'
  elif parts[0]=='private_diagnostic_198':title='198 private prefix'
  elif parts[0]=='drive_diagnostic_246':title='246 link/drive prefix'
  else:title=ALIASES.get(parts[0],parts[0])+(' baseline' if parts[-1]=='baseline' else '')
  item=dict(path=str(p),results=r,trace=trace,ticks=tick,inertial=read(p/'static_inertial_model.json'),geometry=read(p/'geometry.json'),rows=rows,title=title,prefix=float(trace['t'][-1])<19.99);items.append(item)
  for file in ['results.json','trace.npz','every_tick.npz','static_inertial_model.json','geometry.json','manifest.json']:hashes[str(p/file)]=sha(p/file)
  for case,row in rows.items():
   status,reasons=base.label(row);first=[]
   for key,mask in [('slip',~tick['no_slip'][:,case]),('forbidden_contact',tick['wrong'][:,case]>10),('illegal_patch',~tick['legal'][:,case]),('physical_failure',tick['failure'][:,case])]:
    ii=np.flatnonzero(mask)
    if len(ii):first.append(dict(kind=key,time_s=float(tick['t'][ii[0]]),trace_index=int(np.argmin(abs(trace['t']-tick['t'][ii[0]])))))
   episodes.append(dict(source=str(p),relative_source=rel,case=case,title=title,horizon_s=float(trace['t'][-1]),full_duration=not item['prefix'],label='DIAGNOSTIC PREFIX' if item['prefix'] else status,reasons=[] if item['prefix'] else reasons,first_events=first,stable=bool(row['success']) if not item['prefix'] else None,slip_ticks=row['slip_ticks'],illegal_ticks=row['illegal_ticks'],warning_ticks=row['traction_bad_ticks']))
 assert len(items)==41,(len(items),'inventory changed; renderer must be reviewed')
 aborted=[dict(name='24 initial prefix',source=str(D/'sensor_prefix_24'),duration='<=8.4 s; no saved trajectory',reason='aborted diagnostic / no complete trace'),dict(name='24 accounting prefix',source=str(D/'sensor_prefix_v2_24'),duration='8.4 s; no saved trajectory',reason='accounting assertion / no complete trace'),dict(name='24 capture prefix',source=str(D/'sensor_prefix_v3_24'),duration='0.1 s; no saved trajectory',reason='capture-write error / no complete trace')]
 aborted.append(dict(name='196 identity check',source=str(D/'passive_diagnostic_196'),duration='0 active s;600 settling env-ticks',reason='alias-risk check stopped before motion'))
 stopped=read(D/'passive_stopped_status_196.json');assert stopped['active_ticks']==0 and stopped['initial_settling_ticks']==600
 hashes[str(D/'passive_stopped_status_196.json')]=sha(D/'passive_stopped_status_196.json')
 recovery=read(D/'diagnostic_recovery_27.json');hashes[str(D/'diagnostic_recovery_27.json')]=sha(D/'diagnostic_recovery_27.json')
 for a in aborted:assert Path(a['source']).is_dir()
 for it in items:assert np.all(np.isfinite(it['trace']['body']))
 W,H,fps=3840,2160,15;intro=30;outro=30;frames_per_page=34;cw,ch=466,326;pages=[]
 for case in range(24):
  critical={0,199,83,129}
  for ep in episodes:
   if ep['case']==case:
    for ev in ep['first_events']:
     tr=next(i['trace']['t'] for i in items if i['path']==ep['source']);critical.add(int(round(float(tr[ev['trace_index']])/.1))-1)
  assert len(critical)<=frames_per_page,(case,len(critical))
  indices=sorted(set(critical)|set(np.linspace(0,199,frames_per_page).round().astype(int).tolist()))
  while len(indices)>frames_per_page:
   cand=[i for i,x in enumerate(indices) if x not in critical];j=min(cand,key=lambda i:min(indices[i]-indices[i-1] if i else 999,indices[i+1]-indices[i] if i+1<len(indices) else 999));indices.pop(j)
  while len(indices)<frames_per_page:
   _,j=max((b-a,i) for i,(a,b) in enumerate(zip(indices,indices[1:])));indices.insert(j+1,(indices[j]+indices[j+1])//2)
  pages.append(dict(case=case,first_frame=intro+case*frames_per_page,frames=frames_per_page,global_trace_indices=indices,required_indices=sorted(critical)))
 duration=(intro+outro+24*frames_per_page)/fps;assert duration<60
 base.name=lambda key:key
 proc=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s',f'{W}x{H}','-r',str(fps),'-i','-','-an','-c:v','libx264','-threads','2','-preset','fast','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart',str(O/'all_attempts.mp4')],stdin=subprocess.PIPE,stderr=(O/'ffmpeg.log').open('w'))
 def send(im):proc.stdin.write(im.tobytes())
 def title(lines):
  im=Image.new('RGB',(W,H),'#eaf0f3');dr=ImageDraw.Draw(im)
  for i,s in enumerate(lines):dr.text((140,210+i*160),s,font=font(72 if i==0 else 45,i==0),fill='#213a49')
  return im
 hashes[str(Path(__file__).resolve())]=sha(Path(__file__));hashes[str(Path(base.__file__).resolve())]=sha(Path(base.__file__))
 full=[x for x in episodes if x['full_duration']];success=sum(x['label']=='SUCCESS' for x in full)
 opening=title(['Low-seat repair: ALL physical attempts to date',f'{len(full)} complete20s trials: {success} SUCCESS / {len(full)-success} FAILURE','144 recorded diagnostic prefixes;96 aborted prefixes have NO saved motion.','Every source is shown. Labels judge the FULL trial; failures are retained.','Green = clean success; red = failure; amber = margin warning; blue = diagnostic.','Each tile labels its own geometry. Banks differ; do not treat every tile as matched.',f'4K recorded-state schematic / XYZW / {duration:.1f}s / no new PPO or holdout.','Offline endpoint searches are not physical attempts.'])
 for _ in range(intro):send(opening)
 for page in pages:
  case=page['case']
  for gi in page['global_trace_indices']:
   t=(gi+1)*.1;im=Image.new('RGB',(W,H),'#eaf0f3');dr=ImageDraw.Draw(im);dr.text((16,9),f'ALL LOW-SEAT ATTEMPTS | placement row {case+1:02}/24 | recorded t={t:4.1f}s | full-trial outcomes',font=font(30,True),fill='#213a49');dr.text((16,48),'Each tile gives its own height/slope. Schematic from saved transforms; samples include first-event frames. No omitted full-trial failures.',font=font(22),fill='#405a6a')
   for k,it in enumerate(items):
    ti=int(np.argmin(abs(it['trace']['t']-min(t,float(it['trace']['t'][-1])))));card=base.draw_card(it,case,ti,it['title'],cw,ch);cd=ImageDraw.Draw(card);row=it['rows'][case];g=it['geometry'][row['geometry_index']];cd.text((cw-205,46),f'{g["height"]*100:g}cm / {g["pitch_deg"]:+g}deg',font=font(16,True),fill='#213a49')
    if it['prefix']:
     cd.rounded_rectangle((2,2,cw-3,ch-3),radius=10,outline='#356c99',width=5);cd.rectangle((8,36,cw-8,118),fill='white');cd.text((15,40),'DIAGNOSTIC PREFIX',font=font(21,True),fill='#356c99');cd.text((15,72),f'Recorded {it["trace"]["t"][-1]:.1f}s; no full-trial verdict',font=font(16),fill='#356c99');cd.text((15,96),'PREFIX ENDED — last saved state' if t>it['trace']['t'][-1] else 'Measurement diagnostic; not repair validation',font=font(14),fill='#356c99')
    im.paste(card,(16+(k%8)*474,90+(k//8)*338))
   for j,a in enumerate(aborted):
    k=len(items)+j;card=Image.new('RGB',(cw,ch),'#e1e8ee');cd=ImageDraw.Draw(card);cd.rectangle((3,3,cw-4,ch-4),outline='#596d80',width=4)
    for yy,txt,size in [(14,a['name'],22),(53,'ABORTED DIAGNOSTIC',24),(95,f'placement row {case+1}/24',20),(140,a['duration'],20),(185,a['reason'],17),(230,'No motion invented; see coverage/recovery record.',17)]:cd.text((15,yy),txt,font=font(size,yy==53),fill='#41586a')
    im.paste(card,(16+(k%8)*474,90+(k//8)*338))
   send(im)
  print('PAGE',case+1,'/24',flush=True)
 ending=title(['Every recorded attempt included; repair remains unresolved',f'{len(full)} full trials: {success} SUCCESS and {len(full)-success} FAILURE',f'120 recorded prefixes +96 aborted diagnostic prefixes disclosed;{duration:.1f}s total.','Best controller:64/72 clean,70 stable,8 slipping,2 housing contacts.','No new PPO, fresh holdout, policy promotion or relaxed physical criteria.','Torque is approximate implicit-PD telemetry, not hardware motor effort or energy.','See coverage.json and independent verification for source/frame accounting.'])
 for _ in range(outro):send(ending)
 proc.stdin.close();assert proc.wait()==0
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,width,height,r_frame_rate,nb_frames','-of','json',str(O/'all_attempts.mp4')]));assert float(probe['format']['duration'])<60 and probe['streams'][0]['codec_name']=='h264'
 manifest=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='All saved physical trajectories in low_seat_contact_01 through centered-stance probe234;40 source runs,35 complete20s and5 diagnostic prefixes. Every completed failure and success remains included. Four aborted diagnostic attempts without saved trajectories are disclosed as static tiles, never synthesized. Offline model studies had no physical motion. This is an interim video, not repaired/PPO final results.',source_paths=[str(p) for p in paths],source_sha256=hashes,quaternions='XYZW',fps=fps,duration_s=duration,intro_frames=intro,outro_frames=outro,episodes=episodes,aborted_diagnostic_runs=aborted,aborted_diagnostic_case_count=96,pages=pages,probe=probe,physics_ticks=0,optimizer_updates=0,full_trials=len(full),successes=success,failures=len(full)-success,prefixes=120,recovery_record=recovery)
 (O/'coverage.json').write_text(json.dumps(manifest,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(dict(duration_s=duration,full_trials=len(full),successes=success,failures=len(full)-success,prefixes=120)))
if __name__=='__main__':main()
