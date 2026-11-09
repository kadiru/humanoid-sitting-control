"""Report rejected predictive surrogate and independently verified arithmetic."""
from pathlib import Path
import json,hashlib,datetime,csv,shutil,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'progress_report_319'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def table(head,rows):
 return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</table></div>'
def main():
 r=read(D/'ordered_contact_screen_317/results.json');a=read(D/'ordered_contact_audit_318/independent_audit.json');assert a['passed'];O.mkdir(exist_ok=False)
 sel=[x for x in r['rows'] if x['case'] in [0,1] and x['window']==[11.,13.]];labels=[f"{row['case']} / {side}" for row in sel for side in ['left','right']];x=np.arange(4)
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
 for off,order,label,color in [(-.25,None,'Recorded endpoint raw qd','#226e9b'),(0,0,'Drive then ideal contact','#b46a27'),(.25,1,'Ideal contact then drive','#7c579e')]:
  vals=[v for row in sel for v in row['orders'][0 if order is None else order]['actual_roll_median' if order is None else 'predicted_roll_median']]
  ax.bar(x+off,vals,.25,label=label,color=color)
 ax.set_xticks(x,labels);ax.set_ylabel('Median ankle-roll velocity (rad/s)');ax.set_xlabel('Development case / ankle');ax.grid(axis='y',alpha=.2);ax.legend(loc='upper right');ax.set_title('Ordered surrogate does not reproduce saved raw velocities\nFixed 5 ms predictions at 20 times per selected case, 11–13 s')
 fig.savefig(O/'ordered_response.png',dpi=160);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(12,5.2),layout='constrained')
 for o,label,color in [(0,'Drive then ideal contact','#b46a27'),(1,'Ideal contact then drive','#7c579e')]:
  axes[0].bar(np.arange(2)+(o-.5)*.35,[row['orders'][o]['joint_position_increment_rms']*1e6 for row in sel],.35,label=label,color=color)
 axes[0].set_xticks([0,1],['Case 0','Case 1']);axes[0].set_ylabel('All37 joint increment RMS error (µrad)');axes[0].set_title('Position increment agreement also depends on order');axes[0].legend(fontsize=9)
 available=[sum(row['available'] for row in r['rows'] if row['case']==c) for c in range(24)];axes[1].bar(range(24),available,label='Observed floor-only',color='#226e9b');axes[1].bar(range(24),50-np.array(available),bottom=available,label='Incomplete external contacts',color='#b9bec7');axes[1].set_xlabel('Development case');axes[1].set_ylabel('Fixed sampled states');axes[1].set_title('865 eligible / 335 unavailable');axes[1].legend(fontsize=9)
 for ax in axes:ax.grid(axis='y',alpha=.2)
 fig.savefig(O/'position_and_coverage.png',dpi=160);plt.close(fig)
 flat=[]
 for row in r['rows']:
  for order in range(2):
   z=row['orders'][order];flat.append(dict(case=row['case'],window=str(row['window']),samples=row['samples'],available=row['available'],unavailable=row['unavailable'],order=['drive_then_contact','contact_then_drive'][order],joint_velocity_rms=None if z is None else z['joint_velocity_rms'],joint_position_increment_rms=None if z is None else z['joint_position_increment_rms'],left_roll_rms=None if z is None else z['roll_velocity_rms'][0],right_roll_rms=None if z is None else z['roll_velocity_rms'][1]))
 with (O/'all_case_windows.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 tab=table(list(flat[0]),[[f'{v:.6g}' if isinstance(v,float) else ('unavailable' if v is None else v) for v in z.values()] for z in flat])
 lim=''.join('<li>'+html.escape(x)+'</li>' for x in r['scope']['assumptions'])
 src='https://github.com/NVIDIA-Omniverse/PhysX/blob/fc1018a3745664a1db2b95ce03fb5e91eb585f2e/'
 body=f'''<!doctype html><html><head><meta charset="utf-8"><title>Ordered contact model319</title><style>body{{font:18px/1.55 system-ui;max-width:1280px;margin:30px auto;padding:0 24px;color:#203040}}h1{{font-size:34px}}h2{{margin-top:36px}}.notice{{background:#fff2dd;padding:18px;border-left:5px solid #ac6925}}img,video{{max-width:100%}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px}}th,td{{padding:7px;border:1px solid #cbd2dc}}th{{background:#edf1f5}}a{{color:#176798}}</style></head><body>
<h1>An ordered drive/contact model fails the recorded-motion check</h1>
<p class="notice"><b>Physical repair remains unresolved: best verified controller64/72 clean,70 stable,eight slipping,two housing-contact cases.</b> All64 successes remain protected. This round adds zero physics,settling,full20s trials or optimizer updates. PPO and fresh holdout remain unstarted. No controller,criterion,guard,material or solver setting changed.</p>
<h2>Result and scope</h2><p>A fixed local43-coordinate model was evaluated at50 times8.0–12.9s in all24 saved303v2 cases:1,200 records.865 had observed floor-only coverage and were evaluated under both declared orderings;335 remain unavailable because external contacts were incomplete. No order,parameter,gain or delay was fitted. Each prediction covers one5ms interval,with eight modeled position stages and four velocity stages.</p>
<p>For selected cases0/1 at11–13s,the two models predict median ankle-roll velocities of about0.00085–0.00146rad/s; recorded endpoint medians are0.266–0.283rad/s. Ankle-roll prediction RMS errors are0.265–0.285rad/s. The ideal constraints force modeled sole velocities almost to zero,while the recorded velocities do not satisfy those constraints. <b>This surrogate is not a validated command-to-motion predictor and does not justify a control change.</b> Predicted quiet motion is not an actual quiet-guard pass.</p>
<img src="ordered_response.png" alt="Recorded ankle-roll velocity much larger than either ordered surrogate prediction">
<h2>What was modeled</h2><p>Full-body mass,gravity and frozen velocity-bias terms use the preserved44-body inertia construction and measured starting state. Directional Jacobian differences at1µs evaluate the local velocity-bias term offline. Thirty-seven sequential force-drive rows use fixed captured gains,targets,effort limits and zero velocity targets. Only the model’s internal stage ordering changes. Both feet are represented by bilateral six-axis constraints,projected either before or after each drive sweep. Four velocity stages project those constraints with drive updates frozen. These are explicit approximations,not a replica of installed PhysX.</p>
<p>The eight/four stage counts are authored articulation values,not independently captured effective island counts. Saved joint ordering is not certified GPU traversal. No joint-limit,contact-patch friction,CFM or unilateral-contact iteration is reconstructed. Six authored/frozen finger masses still differ by one float32 step; frozen masses remain unchanged. The small position increment error in one ordering does not cure its large raw-velocity error.</p>
<img src="position_and_coverage.png" alt="Joint position increment errors and available external-contact coverage">
<h2>Source trace and independent arithmetic audit</h2><p>The source bundle pins official PhysX tag110.0/5.8.0 at commitfc1018a3745664a1db2b95ce03fb5e91eb585f2e; installed extension110.3.2 is not proven identical. <a href="{src}physx/source/lowleveldynamics/shared/DyCpuGpuArticulation.h#L73">Force-drive coefficients</a> and <a href="{src}physx/source/gpuarticulation/src/CUDA/internalConstraints2.cu#L1737">GPU per-substep and velocity-iteration branches</a> motivate the modeled sequence. With external forces applied each position substep,the public code resets that substep’s applied drive impulse before accumulating it. Velocity iterations do not recompute the drive from final raw qd. <a href="{src}physx/source/gpusolver/src/PxgTGSCudaSolverCore.cpp#L1324">Contact ordering is configurable in the public solver</a>; neither tested simplification is identified as the installed internal path.</p>
<p>Independent318 replayed all1,200 source/mass/contact/bias records and both trajectories for all865 eligible states using recursive body twists,direct55×55 saddle projections and the scalar implicit equation. Maximum mass error{a['maximum_mass_error']:.3e},contact-matrix error{a['maximum_contact_error']:.3e},bias error{a['maximum_bias_error']:.3e},ordered replay error{a['maximum_replay_error']:.3e}. It also independently checked152 aggregate RMS values. These checks verify arithmetic,not model adequacy or physical admission.</p>
<p>The initial source-fetch shell wrapper had a quoting SyntaxError before executing any Python statement; corrected quoting fetched the bundle once. One attempted header URL was absent and is not a source. No screen,audit or simulation was repeated. All prior failed results remain preserved.</p>
<h2>Next design requirement</h2><p>Replacing the contact process by ideal stationary soles is not supported by these records,even with ordered implicit drives. The next useful offline step is to assess finite,unilateral contact constraints and their normal/friction impulse timing against the existing259/303 records and pinned solver code. End-of-step contact returns are not internal iteration impulses. Report unavailable internal quantities explicitly. Do not tune the model to hide this discrepancy,install its predicted velocity,scale273 gains,add joint authority or launch an endpoint. Keep the actual raw-velocity,pitch,gyro,geometry and load guards.</p><ul>{lim}</ul>
<h2>All case/window results</h2>{tab}<p><a href="all_case_windows.csv">All96 ordering/case/window rows</a>; <a href="results.json">full results and scope</a>; <a href="independent_audit.json">independent audit</a>.</p>
<h2>Physical outcomes and effort remain unchanged</h2><p>Latest physical273 is rejected:20 stable,14 clean,eight slipping,two hip-housing/physical failures,zero illegal-patch cases,19 warnings. Selected36cm/+15° cases hover with zero legal seat load,seated hold or quiet release. No-slip hovering is not sitting. No72-case retention followed. Other centered paths still fail3mm clearance. All8 failures must be removed and all64 successes retained in72 matched full20s trials before PPO.</p>
<p>No new effort result. Prior273 vs bestnormal3s on14 common complete clean trials:torque² whole20 +0.0126472%,final10 +0.0264487%,final5 +0.0794278%; strict-both only one trial. These are unselected-success differences,not repair or PPO savings. Approximate implicit-PD telemetry is not solver/hardware motor effort,energy or a global optimum. Failed fragments are excluded.</p>
<h2>Current complete video remains unchanged</h2><p>No new motion. The58.4s schematic includes44 recorded sources:864 full20s trials(503 clean,361 failed),192 diagnostic prefixes and120 aborted case attempts without motion explicitly disclosed.49 panels. Recorded XYZW link poses plus model head COM are not camera footage;0.1s poses bracket0.02s events. Historical totals are not best-controller accuracy.</p><video controls preload="metadata" src="../all_attempts_video_307/all_attempts.mp4"></video></body></html>'''
 # Readability only in text nodes; preserve tags, attributes, code and numeric tokens.
 import re
 prefix,content=body.split('</head>',1)
 parts=re.split('(<[^>]+>)',content)
 for i in range(0,len(parts),2):parts[i]=re.sub(r'(?<=[a-zA-Z,:;])(?=\d)|(?<=\d)(?=[a-zA-Z])',' ',parts[i])
 (O/'findings.html').write_text(prefix+'</head>'+''.join(parts))
 for p,n in [(D/'ordered_contact_screen_317/results.json','results.json'),(D/'ordered_contact_audit_318/independent_audit.json','independent_audit.json'),(D/'ordered_contact_revision_317.json','revision.json'),(D/'ordered_drive_source_316/source_manifest.json','source_manifest.json')]:shutil.copy2(p,O/n)
 (O/'summary.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),repair_complete=False,physical_admission=False,new_physics_ticks=0,video_unchanged=True),indent=2))
 (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2));print('report319 complete')
if __name__=='__main__':main()
