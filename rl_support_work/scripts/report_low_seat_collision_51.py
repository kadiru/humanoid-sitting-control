"""Verified interim collision-model findings; physical repair remains open."""
from pathlib import Path
import json,hashlib,datetime,shutil,html
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from plan_seat_load_targets import clipped_vertices
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');OUT=D/'progress_report_51'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 OUT.mkdir(exist_ok=False)
 manifest=read(D/'cooked_model_50/manifest.json');new=read(D/'cooked_model_50/model.json');old=read(R/'runs/g1_kinematics_geometry_04/model.json')
 assert sha(D/'cooked_model_50/model.json')==manifest['model_sha256']
 for p,h in manifest['source_hashes'].items():assert sha(p)==h,p
 for k in old:
  if k!='collisions':assert old[k]==new[k]
 meshchecks=[]
 for a,b in zip(old['collisions'],new['collisions']):
  if a['type']!='Mesh':assert a==b;continue
  raw=np.asarray(a['points']);cooked=np.asarray(b['points']);assert np.isfinite(cooked).all()
  dist=cKDTree(raw).query(cooked)[0];meshchecks.append(dict(body=a['body'],raw_count=len(raw),cooked_count=len(cooked),max_nearest_raw_point_distance_m=float(dist.max())))
 # Cooking may create vertices, so nearest-raw distance is recorded, not an
 # invented acceptance criterion. Source arrays were exactly matched by query.
 assert len(meshchecks)==32
 query=read(D/'remaining_cooked_meshes_v2_49/results.json');assert query['physics_steps']==query['control_ticks']==query['settling_ticks']==0 and query['timeline_before']==query['timeline_after']==0. and query['timeline_stopped']
 for p,h in query['inputs'].items():assert sha(p)==h
 for p,h in query['reused_queries'].items():assert sha(p)==h
 calc=read(D/'terminal_cooked_screen_50/results.json');oldcalc=read(D/'terminal_screen_42/results.json');model=BodyModel(D/'cooked_model_50/model.json')
 for p,h in calc['source_input_sha256'].items():assert sha(p)==h
 epchecks=[]
 for row in calc['rows']:
  best=row['best_other_screens_feasible'];assert best is not None
  T=transform(Rotation.from_quat(best['root'][3:]).as_matrix(),best['root'][:3]);fk=model.fk(np.asarray(best['q']));sr=Rotation.from_euler('y',row['measured_slope_deg'],degrees=True).as_matrix();center=np.array([-.4,0,row['measured_height_m']]);values=[]
  for name,vertices in model.vertices.items():
   if name in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
   v=(apply(T@fk[name],vertices)-center)@sr;v[:,0]-=.4;v=clipped_vertices(v)
   if len(v):values.append(float(v[:,2].min()))
  error=abs(min(values)-best['clearance_m']);assert error<1e-9
  assert best['all_endpoint_screens_passed']==bool(best['other_screens_passed'] and min(values)>=.003)
  epchecks.append(dict(case=row['case'],assumption=row['assumption'],clearance_replay_error=error,endpoint_screen_passed=best['all_endpoint_screens_passed'],physical_admission=False))
 audit=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,physical_admission=False,all_source_hashes_passed=True,unchanged_joints_inertia_defaults=True,unchanged_primitive_colliders=6,queried_mesh_count=32,mesh_checks=meshchecks,endpoint_checks=epchecks,zero_physics_query_accounting_passed=True,limits='Standalone cooking query, not a live simulation-shape dump. Contact/rest offsets are unavailable; no zero assumption. Six primitives retain prior analytic sampling. No new physical trajectory, contact transfer, success retention or effort measurement.')
 (OUT/'independent_model_audit_51.json').write_text(json.dumps(audit,indent=2))
 sourcefiles={
 'rolling_model_44.json':D/'rolling_contact_screen_v2_44/results.json',
 'arm_authority_45.json':D/'support_authority_45/results.json',
 'independent_model_audit_46.json':D/'model_audit_v2_46/independent_audit.json',
 'recorded_contact_comparison_48.json':D/'cooked_geometry_48/recorded_comparison.json',
 'query_recovery_47.json':D/'cooked_geometry_48/query_recovery_47.json',
 'cooked_model_manifest_50.json':D/'cooked_model_50/manifest.json',
 'corrected_endpoint_screen_50.json':D/'terminal_cooked_screen_50/results.json',
 'query_accounting_49.json':D/'remaining_cooked_meshes_v2_49/results.json',
 'previous_physical_summary_43.json':D/'progress_report_43/summary.json'}
 for name,p in sourcefiles.items():shutil.copy2(p,OUT/name)
 recorded=read(D/'cooked_geometry_48/recorded_comparison.json');curves={r['case']:r for r in recorded['rows'] if r['variant']=='dual_view_approach_40' and r['case'] in [0,2,6,10]}
 fig,axes=plt.subplots(2,4,figsize=(15,7),sharex=True)
 for c,i in enumerate([0,2,6,10]):
  r=curves[i];s=r['samples'];t=[x['t'] for x in s]
  for label,color in [('raw','#888'),('cooked','#128292')]:axes[0,c].plot(t,[x['meshes'][0][label]*1000 if x['meshes'][0][label] is not None else np.nan for x in s],label=label.title()+' pelvis hull',color=color)
  axes[0,c].axhline(0,color='#b34d38',ls='--');axes[0,c].set_ylim(-10,30);axes[0,c].set_title(f"{r['height_cm']:g} cm / +{r['slope_deg']:g}°")
  axes[1,c].plot(t,[x['seat_support_bw'] for x in s],color='#304e8c',label='Recorded legal seat load');axes[1,c].axhline(.05,color='#b34d38',ls='--',label='Existing minimum');axes[1,c].set_ylim(0,.06)
  if r['first_housing_s'] is not None:
   for ax in axes[:,c]:ax.axvline(r['first_housing_s'],color='#b34d38',ls=':')
  for ax in axes[:,c]:ax.set_xlim(10,20);ax.grid(alpha=.2)
  axes[1,c].set_xlabel('Recorded time (s)')
 axes[0,0].set_ylabel('Pelvis gap above finite seat (mm)');axes[1,0].set_ylabel('Legal normal load / bodyweight');axes[0,0].legend(fontsize=8);axes[1,0].legend(fontsize=8)
 fig.suptitle('Raw vertices overstate pelvis reach toward the seat\nRecorded rejected dual40 trials; cooked query is diagnostic, not a new success or sole-cause proof')
 fig.tight_layout();fig.savefig(OUT/'contact_model_comparison.png',dpi=150);plt.close(fig)
 labels=['36 cm / +15°','36 cm / +20°','37 cm / +16.25°','36.5 cm / +17.5°'];cases=[0,2,6,10]
 fig,ax=plt.subplots(figsize=(10,5));x=np.arange(4);width=.25
 for offset,result,assumption,label,color in [(-width,oldcalc,'foot_only_before_contact','Old raw model, foot support','#aaa'),(0,calc,'foot_only_before_contact','Cooked model, foot support','#257d94'),(width,calc,'hypothetical_seat_supported_endpoint','Cooked model, assumed seat support','#b28b38')]:
  values=[next(r for r in result['rows'] if r['case']==i and r['assumption']==assumption)['best_other_screens_feasible']['clearance_m']*1000 for i in cases]
  ax.bar(x+offset,values,width,label=label,color=color)
 ax.axhline(3,color='#b33f3b',ls='--',label='Unchanged model screen: 3 mm');ax.set_xticks(x,labels);ax.set_ylabel('Best tested endpoint housing clearance (mm)');ax.set_ylim(0,8);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
 ax.set_title('Correcting contact geometry narrows the feasible endpoint set\nBounded local searches; no path, contact transition or physical safety certificate')
 fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=150);plt.close(fig)
 summary=dict(utc=audit['utc'],repair_complete=False,task_complete=False,main_training_started=False,fresh_holdout_started=False,policy_promoted=False,new_physical_ticks=0,new_optimizer_updates=0,best_current=dict(clean=64,stable=70,slip=8,housing_contacts=2,trials=72,original_union_retained=60,new_successes_retained=4,warnings=50,strict_clean=22),finding='Queried cooked pelvis contact surface is about5.03–5.95mm above raw-planner hull in the selected recorded late/first-event samples. Raw vertices match configured asset exactly; collision approximation, not a different source mesh, explains this model discrepancy. Recorded force timing is consistent with it, not a single-cause proof. Actual contacts and unsupported outcomes remain failures.',geometry_only=True,recorded_episodes_reviewed=120,queried_meshes=32,primitives_unchanged=6,model50_scope='Separate offline planning collision model; never replace current sensors or visual self-mask with cooked collision points.',no_new_effort_result=True,next='Design a complete support/clearance-feasible approach using the corrected collision representation, including an actual legal load transition and existing40deg terminal support. Deep36/+20 still fails3mm even with hypothetical seat support; do not execute endpoints or repeat blind gain/rate/arm changes. Retain64 and remove8 before PPO.')
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 table=[]
 for r in calc['rows']:
  b=r['best_other_screens_feasible'];label=labels[cases.index(r['case'])]
  table.append(f"<tr><td>{label}</td><td>{html.escape(r['assumption'].replace('_',' '))}</td><td>{np.rad2deg(b['pitch_rad']):.2f}°</td><td>{1000*b['clearance_m']:.3f}</td><td>{'PASS model only' if b['all_endpoint_screens_passed'] else 'FAIL'}</td></tr>")
 event=[]
 for r in recorded['target_summary']:
  s=r['sample'];p=s['meshes'][0];event.append(f"<tr><td>{r['case']}</td><td>{s['t']:.2f}</td><td>{p['raw']*1000:.3f}</td><td>{p['cooked']*1000:.3f}</td><td>{s['seat_support_bw']:.4f}</td><td>{r['first_housing_s'] if r['first_housing_s'] is not None else 'None; still unsupported'}</td></tr>")
 document=f'''<!doctype html><html><head><meta charset="utf-8"><title>Low-seat repair — collision-model diagnosis</title><style>body{{font:16px/1.55 system-ui,sans-serif;max-width:1180px;margin:35px auto;padding:0 22px;color:#182631}}h1,h2{{line-height:1.2}}.warning{{padding:15px 20px;border-left:5px solid #b44;background:#fff2ed}}img{{width:100%;height:auto;border:1px solid #ddd}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{text-align:left;padding:9px;border-bottom:1px solid #ddd}}th{{background:#eef3f6}}code{{background:#eee;padding:2px 4px}}a{{color:#126a8c}}</style></head><body>
 <h1>Collision-model mismatch found; physical repair remains unresolved</h1>
 <p class="warning"><b>Best verified controller: 64/72 clean, 70 stable, 8 slipping, 2 housing-contact cases.</b> All 60 original successes and four gains remain the retention target. This round ran no physical trajectories, optimizer updates or fresh holdout. PPO remains unstarted.</p>
 <h2>What changed in our understanding</h2><p>The raw collision vertices used by the planner match the configured robot asset exactly. PhysX simplifies those meshes into collision hulls. Its queried pelvis hull lies about <b>5.03–5.95 mm farther from the seat</b> than the raw hull at the inspected late/first-event samples. For example, just before the 36 cm / +20° housing event, the raw pelvis appears 2.47 mm inside the plane while the cooked pelvis remains 2.97 mm above it; recorded legal seat load is zero.</p>
 <p>This is a concrete planning-model discrepancy. It is consistent with delayed legal support and does not prove that it alone caused every slip or contact. Housing failures remain real. The previous 0.9-rad terminal posture also still violates the unchanged 40° support criterion. Neither failure is relabeled as success.</p>
 <img src="contact_model_comparison.png" alt="Raw and cooked pelvis gaps with recorded legal load">
 <h2>Corrected model, unchanged requirements</h2><p>A separate offline planning model contains the 32 queried mesh hulls. All joint frames, inertial metadata, defaults and six primitive colliders are preserved. No runtime controller, physics asset, material, threshold, sensor channel or perception self-mask changed. The query created no environment and advanced no physics; its recorded timeline stayed at zero. These are standalone cooked representations, not a live simulation-shape dump. Unauthored contact/rest offsets remain unavailable.</p>
 <img src="comparison.png" alt="Endpoint clearance comparisons for raw and cooked geometry">
 <table><thead><tr><th>Development seat</th><th>Static assumption</th><th>Pitch</th><th>Clearance mm</th><th>Endpoint screen</th></tr></thead><tbody>{''.join(table)}</tbody></table>
 <p>The corrected searches kept the same bounds, fixed feet, previously tested arm pose and 38° research reserve under the existing 40° terminal criterion. The deepest seat fails the unchanged 3 mm model screen even when seat support is hypothetically assumed. Endpoint-only results do not establish a trajectory, actual contact, friction capacity, success or impossibility.</p>
 <h2>Recorded target-event samples</h2><p>All 120 existing episodes were included. The rows below show the eight targets from dual40. Pose samples at 0.1 s only bracket the 0.02 s contact labels; this is not exact-event geometry.</p>
 <table><thead><tr><th>Case</th><th>Sample s</th><th>Raw pelvis gap mm</th><th>Cooked gap mm</th><th>Legal load / BW</th><th>First housing s</th></tr></thead><tbody>{''.join(event)}</tbody></table>
 <h2>Other completed checks and recoveries</h2><p>The evolving-contact rolling screen failed all four whole-path gates: command peaks were 0.207–0.211 rad/s versus its 0.175 bound; two geometries also lost 3 mm clearance. A bounded arm first-moment calculation gained about 18.48 mm of forward COM but still left the deepest raw-model endpoint below 3 mm. Neither is an executable repair. Independent arithmetic, source, mesh and endpoint checks passed as audits only.</p>
 <p>Original failures are preserved: NumPy boolean JSON serialization in 44; a one-sample timestamp mismatch in the first offline audit46; unsupported launcher flag in47; post-shutdown summary write in47; and an assumed mesh path in49. Corrections retained tolerances and reused completed data/queries. No physics rerun or physical criterion change occurred.</p>
 <h2>Effort and remaining work</h2><p>No new physical effort result was measured. The verified09 controller matched current61111 on common clean pairs; against selected62111@8 on 20 common clean old-development trials it used 4.38% more whole-20-second torque squared and 25.45% more final-5-second torque squared. These are existing results, not PPO gains. Margin warnings remain separate: 50 warning cases and 22 strict clean cases in verified09.</p>
 <p>Next is a complete support-feasible contact transition using the corrected collision representation. Preserve all64 current clean conditions and remove all8 targeted failures in matched full20-second development trials before the small two-seed PPO study. Fresh holdout remains closed. No policy promotion. Two seeds and nearby rigid slabs do not establish sitting or leaning anywhere; torque is a proxy, not energy or global optimality.</p>
 <p><a href="summary.json">Summary</a> · <a href="independent_model_audit_51.json">Independent model audit</a> · <a href="recorded_contact_comparison_48.json">All120 recorded episodes</a> · <a href="corrected_endpoint_screen_50.json">Complete bounded search</a> · <a href="query_recovery_47.json">Query recovery</a> · <a href="../progress_report_43/review.html">Previous physical comparison</a></p>
 </body></html>'''
 (OUT/'findings.html').write_text(document)
 (OUT/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in OUT.iterdir() if p.is_file()},indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
