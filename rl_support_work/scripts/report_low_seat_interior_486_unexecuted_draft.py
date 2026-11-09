"""Saved-only report of the completed NEW484 construction and independent485 audit."""
from pathlib import Path
from fractions import Fraction
import json,csv,hashlib,html,datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'progress_report_486'
read=lambda p:json.loads(Path(p).read_text())
def main():
 ex=read(D/'interior_postprocessing_486.json');assert ex['main_actual_exit']==ex['independent_audit_actual_exit']==0
 P=D/'interior_path_484';Q=D/'interior_path_audit_485';result=read(P/'results.json');audit=read(Q/'independent_audit.json');assert audit['passed']
 final=dict(np.load(P/'final.npz'));row=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz'));ind=dict(np.load(Q/'independent_rows.npz'));samples=read(Q/'geometry.json');history=[json.loads(s) for s in (P/'history.jsonl').read_text().splitlines()]
 O.mkdir(exist_ok=False)
 def csvout(name,data):
  with (O/name).open('w',newline='') as f:
   if data:w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
 stages=[];certificates=[];localization=[]
 for h in history:
  for at in [h['primary']]+h['attempts']:
   sub=P/f"linear_{at['linear_index']:03d}";cert=read(sub/'certificate.json') if (sub/'certificate.json').exists() else None
   if cert:certificates.append(dict(linear=at['linear_index'],certificate=cert))
   state=at.get('summary',{});stages.append(dict(linear=at['linear_index'],role=at['role'],trust=h['trust'],status=at['linear']['status'],actual_lp_call=at['linear']['actual_lp_call'],linear_pass=at['linear'].get('linear_pass'),lp_eta=None if cert is None else cert['lp_reported_eta'],exact_eta=None if cert is None else float(Fraction(*cert['certified_relative_clearance'])),certificate_pass=at.get('certificate_pass'),map_pass=at.get('map_pass'),root_pass=at.get('root_pass'),accepted=at['accepted'],soft=state.get('soft_merit'),hard_min=state.get('minimum_hard_row'),phase_min=state.get('minimum_phase'),full_box_min=state.get('coordinate_bound_minimum'),normal=at.get('total_normal_maximum'),hover=at.get('hover_normalized_error'),reason=at.get('rejection','Primary direction is never mapped' if at['role']=='primary' else None)))
   if (sub/'nonlinear_rows.npz').exists():
    rr=np.load(sub/'nonlinear_rows.npz')['rows'];negative=[]
    for g in dict.fromkeys(row['groups'].tolist()):
     ix=np.flatnonzero((row['groups']==g)&row['keep'])
     if len(ix) and np.any(rr[ix]<0):negative.append(dict(group=str(g),negative=int((rr[ix]<0).sum()),minimum_row=int(ix[np.argmin(rr[ix])]),minimum_normalized=float(rr[ix].min())))
    localization.append(dict(linear=at['linear_index'],negative_groups=negative))
 groups=[]
 for g in dict.fromkeys(row['groups'].tolist()):
  ix=np.flatnonzero((row['groups']==g)&row['keep']);rr=final['rows'][ix]
  groups.append(dict(group=str(g),retained=len(ix),negative=int((rr<0).sum()),minimum_normalized=None if not len(ix) else float(rr.min()),minimum_row=None if not len(ix) else int(ix[np.argmin(rr)])))
 csvout('all_stages.csv',stages);csvout('all_1467_samples.csv',samples);csvout('all_constraint_groups.csv',groups)
 csvout('all_331888_buffered_rows.csv',[dict(row=i,group=str(g),retained=bool(row['keep'][i]),normalized=float(final['rows'][i]),scale=float(row['scale'][i])) for i,g in enumerate(row['groups'])])
 def unit(i):
  if i<9120:return 'rad' if i%80<32 else 'Nm' if i%80<64 else 'BW fraction'
  if i<12004:return 'm' if (i-9120)%14<2 else 'rad'
  if i<12212:return 'dimensionless phase'
  if i<15860:return 'rad/segment'
  if i<19508:return 'rad*s'
  if i<19759:return 'm'
  if i==19759:return 's'
  return 'm' if i<20284 else 'rad'
 csvout('all_27708_original_rows.csv',[dict(row=i,unit=unit(i),original=float(x),prior_reserved=float(ind['old_prior'][i])) for i,x in enumerate(ind['old_raw'])])
 for name,obj in [('result.json',result),('audit.json',audit),('certificates.json',certificates),('trial_localization.json',localization),('final_groups.json',groups),('specification.json',read(P/'specification.json')),('exact_final_drop_state.json',read(P/'final_drop_state.json'))]:(O/name).write_text(json.dumps(obj,indent=2))
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
 fig,ax=plt.subplots(2,1,figsize=(11,6));c=[s for s in stages if s['exact_eta'] is not None];t=[s for s in stages if s['soft'] is not None]
 ax[0].plot([s['linear'] for s in c],[s['exact_eta'] for s in c],'o-');ax[0].axhline(0,color='black',ls='--');ax[0].set(xlabel='Interior LP index',ylabel='Exact certified relative clearance')
 ax[1].plot([s['linear'] for s in t],[s['soft'] for s in t],'o-',color='#944737');ax[1].axhline(result['initial']['soft_merit'],ls='--',label='Sealed472 INITIAL');ax[1].legend();ax[1].set(xlabel='Mapped trial index',ylabel='Normalized soft violation');fig.tight_layout();fig.savefig(O/'certificates_and_trials.png',dpi=150);plt.close(fig)
 fig,ax=plt.subplots(2,1,figsize=(11,6));times=np.arange(1,676)*.02
 for axis,values,label in [(ax[0],ind['clock_raw'][:,:32].min(1),'Original rate residual (rad/tick)'),(ax[1],ind['clock_raw'][:,32:].min(1),'Original target headroom (rad)')]:axis.plot(times,values);axis.axhline(0,color='black',ls='--');axis.set(xlabel='Ideal clock time (s)',ylabel=label)
 fig.tight_layout();fig.savefig(O/'exported_clock.png',dpi=150);plt.close(fig)
 esc=lambda v:html.escape(format(v,'.17g') if isinstance(v,float) else str(v))
 def table(data):
  if not data:return '<p>No records in this category.</p>'
  keys=list(data[0]);return '<div class="scroll"><table><tr>'+''.join('<th>'+esc(k)+'</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(x[k])+'</td>' for k in keys)+'</tr>' for x in data)+'</table></div>'
 pairs=[(k,result[k]) for k in ['predictors','assembled_subproblems','actual_lp_calls','primary_lp_assemblies','interior_lp_assemblies','finite_maps','nonlinear_trial_paths','certificate_failures','coordinate_map_failures','accepted_steps','last_accepted_linear_index','stop']]
 pairs += [(k,result['final'][k]) for k in ['hard_pass','monotone_phase_pass','all_inequalities_pass','soft_merit','coordinate_bound_minimum']]
 pairs += [('Actual total-normal error',float(abs(final['equalities'][:-1]).max())),('Actual normalized hover error',float(abs(final['equalities'][-1])))]
 pairs += [(k,audit[k]) for k in ['sampled_geometry_passed','minimum_housing_m','minimum_cop_m','minimum_moving_command_margin_rad','minimum_command_margin_rad','original_clock_rate_minimum','original_clock_target_minimum','new_finite_work_columns']]
 pairs += [(k,audit.get(k,0)) for k in ['inherited479_exact_records','inherited473_exact_records','inherited441_exact_records','duplicate_new_records']]
 selection='No optimization step was accepted. Export X/Y is EXACTLY the inherited sealed472 INITIAL; its completed473 geometry/work is inherited after explicit identity and manifest checks.' if not result['accepted_steps'] else 'The export is the LAST accepted new state under the frozen rule; no rejected or smaller-merit trial was selected.'
 body=f'''<p>Offline development evidence · 482–486 · One 36 cm / +20° placement</p><h1>Exact interior witnesses before path construction</h1><p class="status">{esc(selection)} This is not physical admission. Repair, PPO and fresh holdout remain unfinished.</p>
 <h2>New whole-path formulation</h2><p>The first LP minimizes the existing rate/window feasibility epigraph, with every hard row retained. Its direction is NEVER mapped. A second LP preserves at least half that modeled improvement and maximizes the minimum relative distance from all finite coordinate faces. This includes 11906 full linearized f/p/load intervals,10541 direct p/load intervals and206 drop intervals:22653 intervals. The witness matrix has422107 rows and10541 columns. There is no hard-feasibility slack, new physical allowance or old optimizer continuation.</p>
 <p>Each p/load endpoint is one binary64 value inward. Each independent normal drop reserves its already-proved encoding error U=2^-59m; the exact pivot reserves205U. All original buffers remain. A posteriori exact Fraction dot products certify a strictly positive minimum clearance using unscaled intervals. This measured certificate may differ from the LP-reported eta; the direction itself is never edited. Empty intervals and nonpositive certificates reject. A certified ray must decode with alpha EXACTLY1 and strictly interior p/load/drop values. Exact drop endpoint conservation and the original load-rounding proof remain separate invariants.</p>
 <p>Dependent f is still determined by the actual15-equation root. Exact interval clearance is only a LINEAR MODEL property; all original nonlinear f/p phase,full boxes,support,planning,static-command,normal,hover and signed hard-row checks remain. New states use fresh mechanics/J. The initial state is the EXACT sealed472 INITIAL, inherited without repeating its completed lift. No478 final or rejected witness initializes this run.</p>
 <p>The distinct frozen budget is24 predictors,up to4 interior corrections each,144 LP assemblies/actual-call maximum and120 finite maps maximum;45s per LP. Existing HiGHS1e-9, independent linear1e-7,2^-20 guards,trust floor1/32,quarter correction extent,root10 updates/9 halvings/1e-13 and actual/predicted ratio≥.1 stay. The preliminary predictor chooses a soft ceiling; actual prediction is recalculated on the SAME certified final direction. Positive eta,LP success or an accepted model step never permits execution.</p>
 <h2>Actual outcome and signed gates</h2>{table([dict(metric=k,value=v) for k,v in pairs])}<p>All37 command margin includes unchanged fixed fingers at their original limits; moving-joint margin is listed separately. Original and buffered signed residuals are not rounded away. Different units are not one physical severity ranking, and numerical merit is not physical performance.</p><img src="certificates_and_trials.png" alt="All exact certificates and new nonlinear trial merits"><img src="exported_clock.png" alt="Original ideal-clock rate and target residuals">
 <h2>Every primary,certificate,map and root decision</h2>{table(stages)}<p><a href="certificates.json">Exact rational certificates</a> · <a href="trial_localization.json">Negative groups in every actual nonlinear trial</a>. Primary directions have no implied root or physical attempt. A rejected certificate is not global infeasibility.</p>
 <h2>Independent checks and evidence inheritance</h2><p>Preparation482 and independent483 audited every new matrix coefficient,interval,encoding reserve and exact-certificate case using saved arrays only. Their illustrative half-initial-merit ceiling was an assembly fixture,not a solved predictor result. Independent485 reconstructs every actual-state base and interior assembly,linear witness,exact certificate,finite shared-ray map,nonlinear row/equality decision and final selection. No old preparation or derivative/root fixture is repeated.</p>
 <p>Exact unchanged final states inherit sealed geometry/work only after state,input and manifest identity. Genuinely new final states receive full sampled geometry and central finite-work checks,with detailed arrays saved before aggregate assertions; exact new duplicates reuse their own saved work. The table identifies the scope of each inherited source. A new aggregate error never replaces earlier inherited bounds. Arithmetic audit success does not certify a feasible path or physical support.</p>
 <h2>Complete numerical records</h2><p><a href="all_stages.csv">Every stage</a> · <a href="all_331888_buffered_rows.csv">All331888 buffered rows</a> · <a href="all_27708_original_rows.csv">All27708 original/prior rows with units</a> · <a href="all_1467_samples.csv">All1467 states</a> · <a href="all_constraint_groups.csv">All groups</a> · <a href="result.json">Main result</a> · <a href="audit.json">Independent audit</a> · <a href="specification.json">Frozen specification</a> · <a href="exact_final_drop_state.json">Exact drops</a></p><details><summary>All final constraint groups</summary>{table(groups)}</details><details><summary>All1467 sampled states</summary>{table([{k:s[k] for k in ['record','time_s','phase','housing_m','cop_margin_m','gap_m','moving_command_margin_rad','sampled_geometry_passed']} for s in samples])}</details>
 <h2>Physical repair and effort remain unchanged</h2><p>Best09 remains64/72 clean,70 stable,8 slipping and2 housing contacts. All64 successes are protected;all8 failures remain unresolved. No new settling,full20s episode,prefix,aborted physical attempt,actual float32 forward-filter tick,executed command,policy update,motion or effort result exists. PPO63111/63117 and fresh holdout remain closed. No-slip hovering is not sitting.</p>
 <p>All1467 labels,1460 exact times,1410 shared blocks and45 fixed states remain.650 movement+25 synthetic holds gives13.5s;these holds are NOT25 CURRENT quiet samples. Persistent contact identities,moment COM,left support and original step/lift/arm/endpoint anchors remain. Inherited approach foot-frame drift is not no-slip certification. Finite sampled geometry and linear-command envelopes do not certify continuous geometry or an implicit command curve. Actual float32,full between-sample consistency,both placements,current-input runtime,frozen parent and164/207v2/175/130 guards precede physics.</p>
 <p>All older failures remain preserved:478 has one hard-feasible model step with original rate/target failures;430 retains its prior ideal-polyline pass and separate added-buffer failure. No local solver or certificate failure proves global impossibility. Approximate implicit-PD telemetry is not solver/hardware effort,energy or a global optimum;no failed-hover,prefix or fragment savings are claimed. Reference-assisted rigid-slab simulation does not establish unassisted real-robot or anywhere sitting.</p>
 <h2>Unchanged complete video307</h2><p>58.4s,4K,15fps;44 sources,864 full20s trials(503 clean,361 failed),192 prefixes and120 aborted no-motion attempts. Link-pose/model-COM schematic,not camera footage. No new motion was rendered.</p><video controls preload="metadata" src="../all_attempts_video_307/all_attempts.mp4"></video><p>Automation ACTIVE. Offline completion is not user-task completion.</p>'''
 style='body{font:17px/1.55 system-ui;background:#f4f6f9;color:#243449;margin:0}main{max-width:1120px;margin:auto;padding:32px}h1{font-size:34px;line-height:1.2}h2{margin-top:30px}table{border-collapse:collapse;background:white;font-size:13px;margin:18px 0;width:100%}th,td{text-align:left;vertical-align:top;padding:8px;border-bottom:1px solid #dae1e8;white-space:nowrap}.scroll{overflow-x:auto}img,video{width:100%}.status{padding:18px;background:#fff0e9;border-left:5px solid #a94d40}summary{padding:12px;background:#e5ebf2;cursor:pointer}a{color:#176b93}'
 (O/'findings.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>Interior-witness whole-clock construction484</title><style>'+style+'</style></head><body><main>'+body+'</main></body></html>')
 (O/'summary.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stages=len(stages),accepted_steps=result['accepted_steps'],physical_admission=False,task_complete=False),indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(dict(report=str(O/'findings.html'))),flush=True)
if __name__=='__main__':main()
