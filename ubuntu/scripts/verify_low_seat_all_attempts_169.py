"""Independent saved-result, event-frame and video inventory audit. No physics."""
from pathlib import Path
import json,hashlib,subprocess,datetime
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');V=D/'all_attempts_video_168';O=D/'all_attempts_video_audit_169'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 O.mkdir(exist_ok=False);m=read(V/'coverage.json');checks=[]
 inventory=sorted(str(p.parent) for p in D.glob('**/trace.npz') if (p.parent/'results.json').is_file() and (p.parent/'every_tick.npz').is_file())
 assert sorted(m['source_paths'])==inventory and len(inventory)==35
 for p,h in m['source_sha256'].items():assert sha(p)==h,p
 for p,h in read(V/'SHA256.json').items():assert sha(V/p)==h,p
 assert m['quaternions']=='XYZW' and len(m['pages'])==24
 entries={(x['source'],x['case']):x for x in m['episodes']};assert len(entries)==840
 pages={x['case']:x for x in m['pages']};event_count=0;full=prefix=success=fail=0;contact_examples=[]
 for folder in inventory:
  p=Path(folder);r=read(p/'results.json');tr=np.load(p/'trace.npz');tk=np.load(p/'every_tick.npz');isfull=bool(abs(float(tr['t'][-1])-20)<1e-6);n=tr['body'].shape[1];assert n==24
  assert np.isfinite(tr['body']).all() and np.isfinite(tr['q']).all() and np.isfinite(tr['torque']).all()
  if isfull:assert len(r['episodes'])==24 and len(tk['t'])==1000 and len(tr['t'])==200
  else:
   assert len(r['episodes'])==0
   account=read(p/'diagnostic_accounting.json');assert account['optimizer_updates']==0 and account.get('full20s_episodes',account.get('complete20s_episodes'))==0
   assert abs(float(tr['t'][-1])-account['elapsed_s'])<1e-6
  rows={x['case']:x for x in r['episodes']}
  for c in range(n):
   e=entries[(folder,c)];page=pages[c];indices=page['global_trace_indices'];assert len(indices)==34 and indices==sorted(set(indices)) and indices[0]==0 and indices[-1]==199
   assert page['first_frame']==30+34*c and page['frames']==34
   if isfull:
    row=rows[c];clean=bool(row['success'] and not any(row['failures'].values()) and row['slip_ticks']==0 and row['illegal_ticks']==0)
    assert e['label']==('SUCCESS' if clean else 'FAILURE');assert e['stable']==bool(row['success']);full+=1;success+=int(clean);fail+=int(not clean)
    assert int((~tk['no_slip'][:,c]).sum())==row['slip_ticks']
    assert int((~tk['legal'][:,c]).sum())==row['illegal_ticks']
   else:assert e['label']=='DIAGNOSTIC PREFIX' and e['stable'] is None;prefix+=1
   assert e['full_duration']==isfull and abs(e['horizon_s']-float(tr['t'][-1]))<1e-8
   expected=[]
   for kind,mask in [('slip',np.logical_not(tk['no_slip'][:,c])),('forbidden_contact',tk['wrong'][:,c]>10),('illegal_patch',np.logical_not(tk['legal'][:,c])),('physical_failure',tk['failure'][:,c])]:
    a=np.flatnonzero(mask)
    if not len(a):continue
    t=float(tk['t'][a[0]]);ti=int(np.argmin(np.abs(tr['t']-t)));expected.append(dict(kind=kind,time_s=t,trace_index=ti));gi=int(round(float(tr['t'][ti])*10))-1;assert gi in indices
    vf=page['first_frame']+indices.index(gi);event_count+=1
    if kind=='forbidden_contact':contact_examples.append(dict(source=folder,case=c,event_s=t,recorded_pose_s=float(tr['t'][ti]),video_frame=vf))
   assert e['first_events']==expected
   checks.append(dict(source=folder,case=c,label=e['label'],event_count=len(expected),horizon_s=e['horizon_s']))
 assert (full,prefix,success,fail)==(768,72,m['successes'],m['failures'])
 assert m['aborted_diagnostic_case_count']==72 and len(m['aborted_diagnostic_runs'])==3
 for a in m['aborted_diagnostic_runs']:
  p=Path(a['source']);assert p.is_dir() and not list(p.glob('**/trace.npz'))
 recovery=read(D/'diagnostic_recovery_27.json');assert recovery==m['recovery_record']
 assert recovery['accounting']['prior_original_baseline_active_upper_bound']==10080 and recovery['accounting']['prior_instrumented_baseline_active']==10080 and recovery['accounting']['prior_v3_baseline_active']==120
 probe=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration:stream=codec_name,width,height,r_frame_rate,nb_frames','-of','json',str(V/'all_attempts.mp4')]))
 st=probe['streams'][0];assert st['codec_name']=='h264' and st['width']==3840 and st['height']==2160 and int(st['nb_frames'])==876 and st['r_frame_rate']=='15/1';assert abs(float(probe['format']['duration'])-58.4)<.001
 # Preserve independently extracted representative full-grid frames, including a first forbidden event.
 contact=next(x for x in contact_examples if '/dual_view_approach_40/' in x['source'] and x['case']==2)
 chosen=dict(intro=0,first_contact=contact['video_frame'],known_success=pages[14]['first_frame']+33,retention_end=pages[8]['first_frame']+33,closing=875,new_probe164=pages[0]['first_frame']+33)
 for name,frame in chosen.items():
  subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(V/'all_attempts.mp4'),'-vf',f'select=eq(n\\,{frame})','-frames:v','1',str(O/(name+'.png'))],check=True)
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope='Independent complete source inventory, saved full-trial labels, contact union/slip first-event frame inclusion, prefix/aborted disclosure, H264 duration and source SHA audit. Rendering uses recorded XYZW/body COM schematic; poses bracket 20ms events. Does NOT validate any rejected controller or constitute repair/PPO admission.',source_runs=35,full_trials=full,successes=success,failures=fail,recorded_prefixes=prefix,aborted_prefixes_disclosed=72,event_frames_checked=event_count,checks=checks,contact_example=contact,extracted_frames=chosen,probe=probe,source_sha256={str(Path(__file__).resolve()):sha(__file__),str(V/'coverage.json'):sha(V/'coverage.json'),str(V/'all_attempts.mp4'):sha(V/'all_attempts.mp4')},physics_ticks=0,optimizer_updates=0,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps({k:result[k] for k in ['passed','full_trials','successes','failures','recorded_prefixes','event_frames_checked']}))
if __name__=='__main__':main()
