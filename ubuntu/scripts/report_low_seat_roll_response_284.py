"""Report completed offline response diagnosis; retain failed273 and current276 video."""
from pathlib import Path
import json,hashlib,shutil,datetime,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'progress_report_284'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(headers,rows):return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(h))+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
def main():
 O.mkdir(exist_ok=False);r=read(D/'roll_response_diagnosis_282/results.json');audit=read(D/'roll_response_audit_283/independent_audit.json');assert audit['passed'];a=dict(np.load(D/'roll_response_diagnosis_282/paired_arrays.npz'));ids=r['joint_ids'];t=a['t'];rt=a['raw_t'];src=[D/'roll_response_diagnosis_282/results.json',D/'roll_response_audit_283/independent_audit.json',D/'roll_feedback_probe_273/independent_roll_audit_v2_274.json',D/'all_attempts_video_audit_277/independent_audit.json']
 for i,p in enumerate(src):shutil.copy2(p,O/f'{i:02}_{p.name}')
 fig,ax=plt.subplots(4,2,figsize=(12,11),sharex=True)
 for c in [0,1]:
  for side,j in enumerate(ids):
   color=['#347a8b','#b86a44'][side];label=['Left','Right'][side]
   ax[0,c].plot(rt,a['roll_feedback'][:,c,side]*1000,label=label,color=color)
   ax[1,c].plot(t,(a['q'][1,:,c,j]-a['q'][0,:,c,j])*1000,label=label+' encoder',color=color);ax[1,c].plot(t,(a['command'][1,:,c,j]-a['command'][0,:,c,j])*1000,ls='--',label=label+' command',color=color)
   ax[2,c].plot(rt,a['raw_qd'][1,:,c,j],label=label+'273',color=color);ax[2,c].plot(rt,a['raw_qd'][0,:,c,j],ls='--',label=label+'234',color=color,alpha=.55)
  ax[2,c].axhline(.05,color='black',ls=':',label='Unchanged quiet threshold')
  ax[3,c].plot(t,np.rad2deg(a['body_roll'][1,:,c,0]-a['body_roll'][0,:,c,0]),label='Pelvis',color='#b86a44')
  for side in [0,1]:ax[3,c].plot(t,np.rad2deg(a['body_roll'][1,:,c,side+1]-a['body_roll'][0,:,c,side+1]),label=['Left foot','Right foot'][side])
  for p in ax[:,c]:p.axvline(9.38,ls=':',color='#999');p.axvline(15,ls=':',color='#777');p.grid(alpha=.2);p.legend(fontsize=7);p.set_xlim(8,20)
  ax[0,c].set_title(f'Case {c}: 36 cm / +15°');ax[-1,c].set_xlabel('Recorded time (s)')
 for i,label in enumerate(['Roll correction (mrad)','Paired q / command Δ (mrad)','Raw joint velocity (rad/s)','Paired body roll Δ (°)']):ax[i,0].set_ylabel(label)
 fig.tight_layout();fig.savefig(O/'paired_response.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(2,2,figsize=(12,7),sharex=True)
 for c in [0,1]:
  for b,name in enumerate(['Pelvis','Left foot','Right foot']):ax[0,c].plot(t,(a['body_xyz'][1,:,c,b,1]-a['body_xyz'][0,:,c,b,1])*1000,label=name)
  for si,run in enumerate(['234','273']):
   for side,label in enumerate(['Left','Right']):ax[1,c].plot(t,a['foot_vertices'][si,:,c,side,:,2].min(axis=-1)*1e6,label=label+' '+run,ls=['--','-'][si])
  ax[0,c].set_title(f'Case {c}');ax[1,c].set_xlabel('Recorded time (s)')
  for p in ax[:,c]:p.set_xlim(8,20);p.grid(alpha=.2);p.legend(fontsize=8);p.axvline(15,ls=':',color='#777')
 ax[0,0].set_ylabel('Paired lateral position Δ (mm)');ax[1,0].set_ylabel('Offline cooked foot bottom z (µm)');fig.tight_layout();fig.savefig(O/'stance_geometry.png',dpi=160);plt.close(fig)
 selected=[]
 for row in r['rows']:
  if row['case'] not in [0,1] or row['window']!=[11,14.9]:continue
  selected.append([row['case'],*[f"{j['paired_difference']['q']['median']*1000:.4f}" for j in row['joints']],f"{np.rad2deg(row['bodies'][0]['paired_roll_rad']['median']):.5f}",f"{row['bodies'][0]['paired_xyz_m'][1]['median']*1000:.5f}",*[f"{np.rad2deg(b['paired_roll_rad']['median']):.6f}" for b in row['bodies'][1:]]])
 allrows=[[x['case'],str(x['window']),*[f"{j['paired_difference']['q']['median']*1000:.6g}" for j in x['joints']],f"{np.rad2deg(x['bodies'][0]['paired_roll_rad']['median']):.6g}",f"{x['bodies'][0]['paired_xyz_m'][1]['median']*1000:.6g}"] for x in r['rows']]
 onsets=[[x[k] for k in ['case','roll_feedback_onset_s','first_command_difference_s','first_raw_q_difference_s','first_sampled_body_difference_s','first_sampled_geometry_difference_s']] for x in r['onsets']]
 text=f'''<!doctype html><html><head><meta charset="utf-8"><title>Roll response diagnosis 282–283</title><style>body{{margin:0;background:#f0f5f7;color:#263c48;font:17px/1.55 system-ui}}main{{max-width:1180px;margin:auto;padding:32px}}h1{{line-height:1.2}}h2{{margin-top:32px}}.notice{{background:#fff0e9;border-left:5px solid #b86a44;padding:18px}}img,video{{width:100%;background:white}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:13px;background:white}}td,th{{padding:8px;border-bottom:1px solid #d8e2e6;text-align:left}}th{{background:#dde9ee}}</style></head><body><main>
<h1>The roll correction shifted the pelvis; seated support remains absent</h1><div class="notice"><b>Offline diagnosis only. No new physical attempt or PPO.</b> Best verified controller: 64/72 clean, 70 stable, 8 slipping and 2 housing-contact cases. Probe 273 remains rejected: 20 stable, 14 clean, 8 slipping, 2 housing/physical failures, zero illegal-patch cases and 19 conservative warnings. The two selected hovers never reached legal seat support. No clean gain; all 14 best-probe clean outcomes retained. Full unselected equality remains failed.</div>
<p>This report compares the saved full 20-second trials 234 and 273: all 24 cases, four windows each, all recorded joint/command arrays and both selected raw 20 ms input streams. No controller, material, guard or physical criterion changed. The current <a href="../progress_report_280/findings.html">physical report and complete-clean effort comparisons</a> remain unchanged.</p>
<h2>Measured response in the selected cases</h2><img src="paired_response.png"><p>The new correction begins at the recorded hover tick 9.38 s. During 11–14.9 s, encoder roll angles moved approximately −7.56 to −7.61 mrad relative to 234. Pelvis roll changed about +0.36° and lateral position about −3.1 mm; foot orientation changed only about 0.0004–0.0008°. Raw roll velocity remains far above the unchanged 0.05 rad/s settling requirement. The command affected whole-body posture without producing quiet seated support.</p>
{table(['Case','Left q Δ mrad','Right q Δ mrad','Pelvis roll Δ °','Pelvis lateral Δ mm','Left foot roll Δ °','Right foot roll Δ °'],selected)}
<p>These paired observations do not identify a causal transfer function: feedback is endogenous and other measured feedback evolves with the state. They do not justify scaling the gain, adding an integral, removing the quiet guard or extrapolating a successful controller. Partial-trial torque differences are diagnostic telemetry, not effort savings.</p>
<h2>Foot contact geometry and support coupling</h2><img src="stance_geometry.png"><p>Both feet stay nearly fixed as the pelvis shifts. The offline cooked-foot bottom is around 42–44 µm below the nominal ground plane in these records. This is model geometry from recorded poses, not a certified live collision surface or a replacement for actual contact labels; live contact/rest offsets remain unknown. Previous 259 contact returns and 267 point-velocity analysis remain separate preserved evidence.</p>
<h2>Clock alignment, not an inferred delay</h2><p>A feedback record at tick k uses current inputs at k×0.02 s. The applied-command record is stamped at the end of that control interval, (k+1)×0.02 s. Thus the feedback onset at 9.38 s and command record at 9.40 s do <b>not</b> establish a 20 ms implementation delay. The next raw pose follows that interval; 0.1 s body samples only bracket motion. No timestamp fitting or observation substitution was used.</p>
{table(['Case','Feedback input time s','First differing command end s','First differing raw q s','First differing sampled body s','First differing measured plane s'],onsets)}
<h2>Independent verification</h2><p>Audit 283 passed independent quaternion/component reconstruction of {audit['geometry_component_values']:,} geometry values, all 96 case-window statistics and all 24 onset records. Maximum geometry difference {audit['max_geometry_error']:.3g}; independent sorted-statistic difference {audit['max_sorted_statistic_error']:.3g}. Raw source arrays and hashes match exactly. This verifies the calculation, not physical admission or causality.</p>
<p>The original failed physical and exact-preservation gates remain failed. Approximate implicit-PD applied-torque telemetry is not actual solver or hardware effort. No end-state PD reconstruction or fitted truth wrench was introduced. The previous spatial-wrench feedback proposal remains rejected.</p>
<h2>All 96 case-windows</h2>{table(['Case','Window s','Left q Δ mrad','Right q Δ mrad','Pelvis roll Δ °','Pelvis lateral Δ mm'],allrows)}
<h2>Current complete-attempt video — unchanged</h2><video controls preload="metadata" src="../all_attempts_video_276/all_attempts.mp4"></video><p>58.4 seconds; all 864 full trials (503 clean / 361 failed), 168 diagnostic prefixes and 96 aborted attempts without motion disclosed. Recorded link-pose schematic, not camera footage. No new motion was recorded in 282–284, so the independently verified 276 video is retained without rerendering.</p>
<p>Next research must account for coupled pelvis/foot motion and actual contact transition under the original quiet/support contract. No blind gain, arm, lean or rate sweep, sensor substitution, solver-setting change or endpoint-only launch. All eight remaining cases must be repaired and all 64 successes retained in 72 matched full 20-second development trials before PPO. Fresh holdout remains closed. No policy promotion. Reference-assisted rigid-slab simulation is not real-robot or anywhere-sitting validation; leaning has not broadened.</p></main></body></html>'''
 (O/'findings.html').write_text(text);summary=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,case_windows=96,independent_audit_passed=True,repair_complete=False,physical_admission=False,new_physics_ticks=0,new_optimizer_updates=0,video_unchanged=True,source_sha256={str(p):sha(p) for p in src});(O/'summary.json').write_text(json.dumps(summary,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
