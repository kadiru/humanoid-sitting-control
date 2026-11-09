"""Report offline constitutive compatibility; no new physical/effort/video result."""
from pathlib import Path
import csv,datetime,json,hashlib,shutil,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');S=D/'drive_stage_diagnosis_312';O=D/'progress_report_314'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def f(x):return f'{x:.7g}'
def table(head,rows):
 return '<div class="scroll"><table><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in head)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in r)+'</tr>' for r in rows)+'</table></div>'
def main():
 r=read(S/'results.json');a=read(D/'drive_stage_audit_313/independent_audit.json');assert a['passed'] and not a['physical_admission'];O.mkdir(exist_ok=False)
 selected=r['selected'];lab=[str(x['case'])+' / '+x['joint'].split('_')[0] for x in selected];x=np.arange(4)
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 fig,axes=plt.subplots(1,2,figsize=(14,5.3),layout='constrained')
 for off,key,label,color in [(-.24,'raw_end','Recorded raw qd','#216a9a'),(0,'position_difference_rate','Pose-difference rate (offline)','#b46a26'),(.24,'required_conditional_velocity','Algebraic conditional variable','#8b4fa3')]:
  axes[0].bar(x+off,[z[key]['median'] for z in selected],.24,label=label,color=color)
 axes[0].set_xticks(x,lab);axes[0].set_ylabel('Median (rad/s)');axes[0].set_title('Three distinct quantities, 11–13 s');axes[0].legend(fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.14))
 for off,key,label,color in [(-.18,'raw_end_PD_residual','Endpoint raw-qd PD','#216a9a'),(.18,'pose_difference_PD_residual','Midpoint / pose-difference PD','#b46a26')]:
  axes[1].bar(x+off,[z[key]['rms'] for z in selected],.36,label=label,color=color)
 axes[1].set_xticks(x,lab);axes[1].set_ylabel('RMS(expression − transmitted moment), Nm');axes[1].set_title('Neither expression reproduces joint-wrench data');axes[1].legend(fontsize=9,loc='upper center',bbox_to_anchor=(.5,-.14))
 for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_xlabel('Case / ankle side')
 fig.suptitle('Offline check only — no alternate velocity or controller admitted')
 fig.savefig(O/'drive_quantities.png',dpi=160);plt.close(fig)
 arr=dict(np.load(S/'arrays.npz'));meta=read(D/'incoming_diagnostic_v2_303/actuator_metadata.json')
 fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
 for col,side in enumerate(['left','right']):
  ji=meta['joint_names'].index(side+'_ankle_roll_joint')
  for c,color in [(0,'#216a9a'),(1,'#b46a26')]:
   axes[0,col].plot(arr['t1'],arr['required_conditional_velocity'][:,c,ji],label=f'Case {c}, conditional',color=color)
   axes[0,col].plot(arr['t1'],arr['required_with_armature_term'][:,c,ji],color=color,alpha=.45,lw=.7,ls='--',label=f'Case {c}, armature variant')
   axes[1,col].plot(arr['t1'],arr['pose_difference_PD'][:,c,ji]-arr['transmitted_axial'][:,c,ji],color=color,lw=.8,label=f'Case {c}')
  axes[0,col].set_title(side.title()+' ankle roll');axes[0,col].set_ylabel('Conditional variable (rad/s)');axes[0,col].legend(fontsize=8)
  axes[1,col].set_ylabel('Pose-difference PD residual (Nm)');axes[1,col].set_xlabel('Recorded time (s)')
  for ax in axes[:,col]:ax.grid(alpha=.2)
 fig.suptitle('Unobserved algebraic variable; separate armature term is not a calibrated rotor model')
 fig.savefig(O/'stage_traces.png',dpi=160);plt.close(fig)
 keys=['raw_end','position_difference_rate','required_conditional_velocity','required_with_armature_term','raw_end_PD_residual','pose_difference_PD_residual']
 def row(z):return [z['case'],z['joint'],z['window']]+[f(z[k]['rms' if 'residual' in k else 'median']) for k in keys]+[f(z['minimum_authored_limit_margin_rad'])]
 headers=['Case','Joint','Window','Raw qd median','Pose rate median','Conditional median','Armature variant median','Raw PD residual RMS Nm','Pose PD residual RMS Nm','Limit margin rad']
 sel_table=table(headers,[row(z) for z in selected])
 joint_table=table(headers,[row(z) for z in r['rows'] if z['case'] in [0,1] and z['window']==[11.,13.]])
 ankle_table=table(headers,[row(z) for z in r['rows'] if 'ankle_roll' in z['joint']])
 flat=[]
 for z in r['rows']:
  zz={}
  for k,v in z.items():
   if isinstance(v,dict):zz.update({k+'_'+kk:vv for kk,vv in v.items()})
   else:zz[k]=v
  flat.append(zz)
 with (O/'all_1776_case_joint_windows.csv').open('w') as out:
  w=csv.DictWriter(out,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
 limits=''.join('<li>'+html.escape(v)+'</li>' for v in r['limitations'])
 body=f'''<!doctype html><html><head><meta charset="utf-8"><title>Drive stage compatibility314</title><style>body{{font:18px/1.55 system-ui;max-width:1360px;margin:30px auto;padding:0 24px;color:#203040}}h1{{font-size:34px}}h2{{margin-top:38px}}.notice{{background:#fff2dd;padding:18px;border-left:5px solid #ac6925}}img,video{{max-width:100%}}.scroll{{overflow:auto}}table{{border-collapse:collapse;font-size:13px}}th,td{{padding:7px;border:1px solid #cbd2dc}}th{{background:#edf1f5}}a{{color:#176798}}code{{font-size:15px}}</style></head><body>
<h1>Drive-stage compatibility from saved measurements</h1>
<p class="notice"><b>Repair remains unresolved: 64/72 clean, 70 stable, eight slipping and two housing-contact cases.</b> This round adds zero physics, settling, full-duration trials or optimizer updates. All 64 current successes remain protected. PPO and fresh holdout are unstarted. No physical criteria, controller guards, materials or gains changed.</p>
<h2>What this check establishes</h2><p>All 1,000 saved 5 ms intervals × 24 development cases × 37 joints are included: 888,000 joint-interval records and 1,776 case/joint/window summaries. The source is verified passive diagnostic 303v2, which exactly replayed the original 234 trajectory. Each prelaunch target is checked against the corresponding backend target, with fixed post-step alignment. Incoming joint torque is projected onto its free revolute axis at the child-joint origin. It describes total transmitted axial moment, not isolated motor effort.</p>
<p>For the two selected cases at 11–13 s, raw ankle-roll velocities have medians about 0.266–0.284 rad/s. The midpoint-position/pose-difference PD expression still disagrees with transmitted moment by 0.144–0.156 Nm RMS; using raw endpoint velocity disagrees by 0.921–0.992 Nm. <b>Neither comparison validates a simple PD force model for this recorded quantity.</b></p>
<img src="drive_quantities.png" alt="Distinct raw velocity, pose-difference and conditional quantities, and torque-expression residuals">{sel_table}
<h2>The conditional variable is not a sensor</h2><p>For diagnosis only, solve the hypothetical scalar identity <code>tau = Kp (target − q_mid) + Kd (v_target − v_conditional)</code> for <code>v_conditional</code>. Its selected late medians are about 0.0360–0.0386 rad/s. This is an algebraic result conditional on an unverified identification of total transmitted moment with drive moment. It is <b>not measured internal solver velocity, a fitted estimator, or a quiet-guard pass</b>. Its numerical size relative to the 0.05 rad/s guard grants no permission to substitute it. Raw qd remains the runtime quantity.</p>
<p>The separately reported armature variant adds the configured armature times the change in raw qd divided by the interval. It is descriptive, not a certified effective rotor model or correction. Limit margins, effort-limit exceedance counts and observed floor-only coverage are retained for every row. Non-foot contacts do not disappear from the original labels.</p><img src="stage_traces.png" alt="Conditional quantities and pose-difference expression residuals through time">
<h2>Independent audit and next design constraint</h2><p>Audit 313 independently rotates child-joint moments and axes into world coordinates, projects them, checks both defining identities, and recomputes all summaries with sorted medians and scalar sums. All 888,000 records and 1,776 windows pass {a['checks']:,} checks; maximum arithmetic difference {a['max_error']:.3e} under the unchanged 1e-10 arithmetic bound. This is arithmetic verification, not physical admission.</p>
<p><a href="https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/_downloads/6acf3afb8f69452757e0e766b5a22978/implicitDrives.pdf">NVIDIA’s implicit-drive note</a> distinguishes TGS position-substep accumulation from the later velocity iterations. End-of-step observations do not reveal the full internal sequence. Authored iteration counts and force-drive attributes are not independently captured effective solver-island counts or backend drive type. No solver settings changed.</p>
<p>A further control design must account for contact-constrained motion and the implicit drive update, with a defensible command-to-motion response. Neither the earlier ideal-contact backward-Euler derivative nor a substitution of pose-difference velocity is validated here. No gain scaling, fitted truth offset, extra joint authority or physical launch follows this audit. Preserve the approach balance, bounded pitch bias, full path, current input validity, total authority, filter/rate and quiet/load guards.</p><ul>{limits}</ul>
<h2>Selected cases, all 37 joints</h2>{joint_table}
<h2>All cases, both ankle-roll joints and both windows</h2>{ankle_table}<p><a href="all_1776_case_joint_windows.csv">All 1,776 case/joint/window records (CSV)</a> and <a href="results.json">full source-linked results</a>.</p>
<h2>Physical and effort status</h2><p>Latest full 20 s probe 273 remains rejected: 20 stable, 14 clean, eight slipping, two hip-housing/physical failures, zero illegal-patch cases, 19 warnings. Selected rows 0/1 hover with zero legal seat load, seated hold or quiet release. No-slip hovering is not sitting. The remaining other centered paths still fail their 3 mm model clearance screen. All eight failures must be removed and all 64 current successes retained in 72 matched full 20 s development trials before PPO.</p>
<p>No new effort result. Prior 273 versus best normal3s on 14 common complete clean trials: torque² whole20 +0.0126472%, final10 +0.0264487%, final5 +0.0794278%; strict-both only one trial. These are unselected-success differences, not repair or PPO savings. Approximate implicit-PD telemetry is a proxy, not actual motor effort or energy. No failed-prefix/hover/fragment savings.</p>
<h2>Current all-attempt video, unchanged</h2><p>No new physical motion this round. The current 58.4 s schematic still includes 44 recorded sources: 864 complete trials (503 clean, 361 failed), 192 diagnostic prefixes and 120 aborted case attempts without saved motion explicitly disclosed. All failures retained; historical repeated totals are not controller accuracy. Recorded XYZW link poses plus model head COM are not camera footage; 0.1 s poses bracket 0.02 s labels.</p><video controls preload="metadata" src="../all_attempts_video_307/all_attempts.mp4"></video>
</body></html>'''
 (O/'findings.html').write_text(body)
 for src,name in [(S/'results.json','results.json'),(D/'drive_stage_audit_313/independent_audit.json','independent_audit.json'),(D/'drive_stage_revision_312.json','revision.json'),(D/'drive_iteration_context_310/context.json','primary_context.json')]:shutil.copy2(src,O/name)
 (O/'summary.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sha(__file__),new_physics_ticks=0,new_optimizer_updates=0,repair_complete=False,physical_admission=False,video_unchanged=True),indent=2))
 (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir()},indent=2));print('report314 complete',flush=True)
if __name__=='__main__':main()
