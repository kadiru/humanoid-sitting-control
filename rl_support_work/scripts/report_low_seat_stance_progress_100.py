"""Report completed offline95/96/97 evidence; explicitly label98 pending."""
from pathlib import Path
import datetime, hashlib, html, json, shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
O=D/'progress_report_100'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    audit=read(D/'landing_ankle_path_audit_96/independent_audit.json');assert audit['passed']
    old=read(D/'contact_path_screen_v2_91/results.json')['dense']
    data=read(D/'landing_ankle_path_95/results.json');z=data['dense']
    diagnosis=read(D/'stance_support_diagnosis_97/diagnosis.json')
    assert not (D/'stance_com_path_98/results.json').exists(), 'Use a new completed-results report if98 has finished.'
    O.mkdir(exist_ok=False)
    fig,axes=plt.subplots(2,2,figsize=(13,7.8))
    specs=[('foot_position_error_m',1000,3,'Foot position error (mm)'),
           ('foot_orientation_error_rad',180/np.pi,.5,'Foot orientation error (degrees)'),
           ('clearance_m',1000,3,'Housing clearance (mm)'),
           ('min_command_margin_rad',1,.03,'Command joint margin (rad)')]
    for ax,(key,scale,limit,label) in zip(axes.flat,specs):
        for source,name,color in [(old,'Path91','#8b9aa1'),(z,'Pitch-shaped95','#266d82')]:
            rows=source['command_nodes'] if key=='min_command_margin_rad' else source['path']
            ax.plot(np.arange(len(rows)),[v.get(key,np.nan)*scale for v in rows],label=name,color=color)
        ax.axhline(limit,color='#b3423c',ls='--',label='Unchanged model limit')
        for boundary in [16,64,104]:ax.axvline(boundary,color='#999',ls=':',alpha=.5)
        ax.set_ylabel(label);ax.set_xlabel('Offline path knot');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=9);axes[1,0].set_ylim(0,15)
    fig.suptitle('Offline landing correction: model constraints remain binding')
    fig.tight_layout(rect=(0,.02,1,.96));fig.savefig(O/'landing_constraints.png',dpi=160);plt.close(fig)
    r=diagnosis['records'][1];v=np.asarray(r['left_sole_xy']);h=ConvexHull(v);p=v[np.r_[h.vertices,h.vertices[0]]]
    fig,ax=plt.subplots(figsize=(10,5.5));ax.plot(p[:,0]*1000,p[:,1]*1000,color='#617c87',label='Modeled left sole')
    target=np.array(r['fixed_stance_target_xy']);a,b=r['sagittal_interval_at_fixed_y_with_5_5mm_reserve']
    ax.plot([a*1000,b*1000],[target[1]*1000]*2,color='#287f67',lw=5,label='Same5.5mm inset at fixed lateral target')
    ax.scatter(* (target*1000),s=100,color='#ad4b3e',label='Original stance target',zorder=4)
    feasible=[v for v in r['target_only_checks'] if v['inside_existing_target_reserve']]
    ax.scatter([v['target_xy'][0]*1000 for v in feasible],[v['target_xy'][1]*1000 for v in feasible],s=32,color='#244c81',label='Diagnostic target locations')
    ax.set_aspect('equal');ax.set_xlabel('Forward coordinate (mm)');ax.set_ylabel('Left coordinate (mm)');ax.grid(alpha=.2)
    ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=9);ax.set_title('Unused stance area is a planning hypothesis, not validated support')
    fig.tight_layout();fig.savefig(O/'stance_support.png',dpi=160);plt.close(fig)
    compact={k:v for k,v in z.items() if k not in ['path','command_nodes']}
    summary=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),task_complete=False,
        repair_complete=False,best_verified=dict(clean=64,total=72,stable=70,slipping=8,housing_contacts=2,conservative_warnings=50,strict_clean=22),
        completed95=compact,search_evaluations=data['search']['evaluations'],audit96_passed=True,
        pending98='Bounded same-sole sagittal COM planning with unchanged endpoint and limits; no result at report cutoff.',
        physical_ticks=0,optimizer_updates=0,holdout_started=False,policy_promoted=False,video_current=True)
    (O/'summary.json').write_text(json.dumps(summary,indent=2))
    for dest,source in [('independent_audit96.json',D/'landing_ankle_path_audit_96/independent_audit.json'),
                        ('stance_diagnosis97.json',D/'stance_support_diagnosis_97/diagnosis.json'),
                        ('stance_revision98.json',D/'stance_com_revision_98.json'),
                        ('video_audit.json',D/'all_attempts_video_audit_60/independent_audit.json')]:shutil.copy2(source,O/dest)
    rows=[('Dense path geometry',z['model_geometry_passed']),('Static command bounds',z['model_command_passed']),
          ('Specific command-time screen',z['command_schedule_passed']),('Failed geometry knots',str(z['failed_knots'])),
          ('Minimum housing clearance',f"{z['min_clearance_m']*1000:.4f} mm"),
          ('Maximum foot position error',f"{z['max_foot_position_error_m']*1000:.4f} mm"),
          ('Maximum foot orientation error',f"{np.rad2deg(z['max_foot_orientation_error_rad']):.5f}°"),
          ('Minimum command margin',f"{z['min_command_margin_rad']:.7f} rad (requires0.03)"),
          ('Time for this command polyline at0.15rad/s',f"{z['minimum_time_for_this_command_polyline_at015_s']:.5f} s (allowance13.5s)"),
          ('Actual filtering, sensed load and dynamic stability','Not validated')]
    table=''.join('<tr><td>'+html.escape(str(a))+'</td><td>'+html.escape(str(b))+'</td></tr>' for a,b in rows)
    page=f'''<!doctype html><html><head><meta charset="utf-8"><title>Low-seat repair: landing and stance check</title>
    <style>body{{font:17px/1.6 system-ui;background:#eff3f4;color:#203743;margin:0}}main{{max-width:1120px;margin:auto;padding:34px}}h1{{font-size:34px;line-height:1.25}}.notice{{background:#fff;border-left:5px solid #b84c42;padding:22px}}img,video{{width:100%;background:#fff}}table{{background:white;border-collapse:collapse;width:100%}}td,th{{padding:12px;text-align:left;border-bottom:1px solid #d2dce0}}a{{color:#176580}}</style></head>
    <body><main><h1>The physical repair is still unresolved</h1><div class="notice"><b>Best verified:64/72 clean,70 stable,8 slipping,2 housing-contact cases.</b><p>All64 successes remain protected. No new physical trial, PPO update or fresh holdout ran in this round. Fifty conservative warning cases and22 strict-clean cases remain separate from physical outcomes.</p></div>
    <h2>Completed landing-path search95</h2><p>The bounded search tested{data['search']['evaluations']} coarse candidates, then one115-knot path. It varied a temporary pitch term to address the recorded right-ankle limit while preserving the static contact endpoint. The independent audit replayed every dense knot, fixed commands, force equilibrium and joint work. Audit success confirms the calculations; it does not convert a failed model gate into admission.</p>
    <img src="landing_constraints.png" alt="Unchanged geometry and command constraints for paths91 and95"><table><tr><th>Offline check</th><th>Result</th></tr>{table}</table>
    <h2>Next bounded hypothesis: use the stance area already available</h2><p>The old path kept its fore–aft COM target fixed while the right foot moved140mm backward. Recorded ankle saturation occurs as that foot returns flat. Diagnosis97 finds{r['available_rearward_shift_m']*1000:.2f}mm of rearward target range inside the same left sole and existing5.5mm reserve. A point inside that polygon is not a dynamic-support certificate.</p>
    <img src="stance_support.png" alt="Measured-model left stance sole and available sagittal target range">
    <p><b>Search98 is pending at this report cutoff.</b> It tests a rearward target shift of at most80mm, further limited by that inset, while retaining the endpoint, foot-lift path, fixed arm targets, joint and command bounds, effort limits,0.15rad/s cap and terminal criteria. It does not move the planted foot. The existing crouch/roll variables keep their prior bounds. Maximum40 coarse evaluations and one dense check; an independent audit follows. No automatic physical or PPO launch.</p>
    <h2>All recorded attempts in58.4seconds</h2><video controls preload="metadata" src="../all_attempts_video_v2_59/all_attempts.mp4"></video><p>The video remains current:696 complete physical trials (405successes,291failures),72 recorded diagnostic prefixes and explicit disclosure of72 aborted case attempts without saved motion. These aggregate historical counts are not current-controller accuracy. This is an offline schematic of recorded XYZW/body transforms;0.1s samples bracket0.02s event labels. Independent coverage checks462 first-event samples.</p>
    <h2>Physical validation and PPO still remain</h2><p>All eight failures must be removed and all64 current successes retained across72 matched full20s development trials before the small two-seed PPO comparison. Fresh holdout remains closed. Materials, slip and contact criteria, terminal tilt≤40° and legal seat normal load≥0.05BW are unchanged. Modeled seat load is not sensed support.</p><p>No new physical effort result: verified09 matches61111 on common complete clean trials; versus selected62111 at8s on20 old-development common-clean trials, whole20 torque squared is4.38% higher and final5 is25.45% higher. Failed fragments are excluded. Torque is a proxy, not energy.</p><p>This remains reference-assisted rigid-slab simulation, not a demonstration of unassisted sitting, real-robot performance, or sitting and leaning anywhere. Leaning has not broadened.</p>
    <p><a href="summary.json">Summary</a> · <a href="independent_audit96.json">Independent audit96</a> · <a href="stance_diagnosis97.json">Stance diagnosis97</a> · <a href="stance_revision98.json">Bounded revision98</a> · <a href="video_audit.json">Video coverage</a> · <a href="SHA256.json">Hashes</a></p></main></body></html>'''
    (O/'findings.html').write_text(page)
    (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
    print(json.dumps(dict(report=str(O),entries=len(read(O/'SHA256.json')),completed95=compact)))
if __name__=='__main__':main()
