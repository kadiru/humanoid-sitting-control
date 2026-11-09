"""Report completed offline98–109; no new physical outcomes."""
from pathlib import Path
import datetime,hashlib,html,json,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'progress_report_110'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    O.mkdir(exist_ok=False)
    z=read(D/'stance_com_path_v2_98/results.json')['dense']
    force=read(D/'contact_load_sharing_106/results.json')
    authority=read(D/'contact_force_authority_104/results.json')
    a107=read(D/'contact_load_filter_audit_107/independent_audit.json')
    schedule=read(D/'contact_filtered_schedule_108/results.json')
    for path in ['stance_com_path_audit_v2_99','contact_force_authority_audit_105',
                 'contact_load_filter_audit_107','contact_filtered_schedule_audit_109']:
        assert read(D/path/'independent_audit.json')['passed']
    fig,ax=plt.subplots(2,1,figsize=(12,8),sharex=True)
    x=np.arange(115)
    ax[0].plot(x,[r['min_command_margin_rad'] for r in z['command_nodes']],label='Path98 prescribed load split',color='#ad654a')
    ax[0].plot(x,[r['best_margin_rad'] for r in authority['nodes']],label='Best margin with same prescribed split (104)',color='#3b6f9c',ls=':')
    ax[0].plot(x,[r['command_margin_rad'] for r in force['nodes']],label='Revised double-support split (106)',color='#23745c')
    ax[0].axhline(.03,color='#a83340',ls='--',label='Unchanged command margin: 0.03 rad')
    ax[0].set_ylim(.0285,.035);ax[0].set_ylabel('Command margin (rad)');ax[0].legend(fontsize=9,loc='upper left')
    ax[0].set_title('Late-approach detail: the fixed-load command failure is real')
    ax[1].plot(x,[v['planned_right_load_fraction'] for v in z['command_nodes']],color='#ad654a',label='Original right-foot load')
    ax[1].plot(x,[v['planned_right_load_fraction'] for v in force['nodes']],color='#23745c',label='Revised right-foot load')
    ax[1].plot(x,[v['assumed_seat_normal_fraction'] for v in force['nodes']],color='#685c89',ls=':',label='Assumed seat normal load')
    ax[1].set_ylabel('Fraction of body weight');ax[1].set_xlabel('Offline path knot');ax[1].legend(fontsize=9)
    for a in ax:a.grid(alpha=.2);a.set_xlim(88,114);a.axvline(104,color='#aaa',ls=':')
    fig.tight_layout();fig.savefig(O/'force_constraints.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(1,2,figsize=(12,5.4))
    labels=['Path98\ncommand polyline','Revised106\ncommand polyline','Constructed108\nfiltered schedule']
    times=[z['minimum_time_for_this_command_polyline_at015_s'],force['summary']['minimum_command_polyline_time_s'],schedule['duration_s']]
    bars=ax[0].bar(labels,times,color=['#8c9da4','#4b8593','#b6714e']);ax[0].axhline(13.5,color='#a83340',ls='--',label='Existing 13.5 s allowance')
    for b,t in zip(bars,times):ax[0].text(b.get_x()+b.get_width()/2,t+.4,f'{t:.2f} s',ha='center',fontsize=11)
    ax[0].set_ylim(0,29);ax[0].set_ylabel('Offline command time (seconds)');ax[0].legend(fontsize=9)
    f=np.load(D/'contact_filtered_schedule_108/command_schedule.npz')
    ax[1].plot(f['t'],f['source_arc']/f['source_arc'][-1],color='#4b8593')
    ax[1].axvline(13.5,color='#a83340',ls='--');ax[1].set_xlabel('Command-only time (seconds)');ax[1].set_ylabel('Progress along unchanged command polyline')
    ax[1].set_title('No clipped targets; time gate still fails');ax[1].grid(alpha=.2)
    fig.suptitle('No physical motion or minimum-time proof in these calculations');fig.tight_layout(rect=(0,0,1,.94));fig.savefig(O/'filter_timing.png',dpi=160);plt.close(fig)
    compact={k:v for k,v in z.items() if k not in ['path','command_nodes']}
    summary=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),task_complete=False,repair_complete=False,
                 best_verified=dict(clean=64,total=72,stable=70,slipping=8,housing_contacts=2,conservative_warnings=50,strict_clean=22),
                 completed98=compact,force102=read(D/'contact_force_schedule_102/results.json')['summary'],
                 authority104=authority['summary'],load_sharing106=force['summary'],filter107=a107['command_only_filter'],schedule108=schedule,
                 physics_ticks=0,optimizer_updates=0,holdout_started=False,policy_promoted=False,video_current=True,
                 conditional103='Not run:102 was infeasible and produced no command path; original unexecuted source preserved.')
    (O/'summary.json').write_text(json.dumps(summary,indent=2))
    files={'independent_audit99.json':'stance_com_path_audit_v2_99/independent_audit.json',
           'force_authority104.json':'contact_force_authority_104/results.json',
           'independent_audit105.json':'contact_force_authority_audit_105/independent_audit.json',
           'independent_audit107.json':'contact_load_filter_audit_107/independent_audit.json',
           'filtered_schedule108.json':'contact_filtered_schedule_108/results.json',
           'independent_audit109.json':'contact_filtered_schedule_audit_109/independent_audit.json',
           'load_revision106.json':'load_sharing_revision_106.json',
           'video_audit.json':'all_attempts_video_audit_60/independent_audit.json'}
    for dest,src in files.items():shutil.copy2(D/src,O/dest)
    checks=[('Path98 sampled geometry','PASS: all 115 knots; housing clearance 3.7311 mm'),
            ('Path98 original commands and time','FAIL: margin 0.0294052 rad; command-polyline time 15.1613 s'),
            ('Fixed-load force redistribution102','INFEASIBLE; no command path produced'),
            ('Independent authority104/105','Knots 99 and 100 still fail the actual 0.03 rad margin'),
            ('Revised double-support allocation106','Static command limits PASS; minimum margin 0.0300100 rad'),
            ('106 specific polyline time','FAIL: 14.5080 s at unchanged 0.15 rad/s; allowance 13.5 s'),
            ('107 unchanged-filter check','FAIL: 88 clipped inverse targets; max mismatch 0.005193 rad'),
            ('108 constructed slower schedule','Arithmetic PASS, time FAIL: 25.10 s; 1,255 command ticks, zero physics ticks'),
            ('109 independent replay','Exact Torch CPU applied commands, fixed fingers and endpoint; zero clipped targets'),
            ('Sensed support and dynamic stability','NOT VALIDATED; all forces here are assumed static loads')]
    table=''.join(f'<tr><td>{html.escape(a)}</td><td>{html.escape(b)}</td></tr>' for a,b in checks)
    phase=''.join(f'<tr><td>{html.escape(k)}</td><td>{v:.2f} s</td></tr>' for k,v in schedule['phase_time_s'].items())
    page=f'''<!doctype html><html><head><meta charset="utf-8"><title>Low-seat repair — force and filter checks</title>
    <style>body{{font:17px/1.55 system-ui;background:#eef2f3;color:#243842;margin:0}}main{{max-width:1120px;margin:auto;padding:36px}}h1{{font-size:34px;line-height:1.2}}h2{{margin-top:34px}}.notice{{background:white;border-left:5px solid #b65c45;padding:22px}}img,video{{width:100%;background:white}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:12px;border-bottom:1px solid #d1dadd;text-align:left}}a{{color:#166a80}}</style></head><body><main>
    <h1>The physical repair remains unresolved</h1><div class="notice"><b>Best verified result: 64/72 clean, 70 stable, 8 slipping, 2 housing-contact cases.</b><p>All 64 successes remain protected. This round added no physical trials, optimizer updates or fresh-holdout tests. Fifty conservative warning cases and 22 strict-clean cases remain separate.</p></div>
    <h2>A specific command constraint, then a time constraint</h2><p>Stance-path98 passed all sampled geometry checks but failed static command bounds and timing. Force-only redistribution102 was infeasible. Independent finite-difference work and saved primal/dual witnesses104/105 locate a real left-hip-roll command deficit at late-approach knots99/100, even at the actual 0.03 rad limit. This is a result for these fixed poses and prescribed loads, not a proof that other motions are impossible.</p>
    <p>Revision106 changes only the internal modeled load split after both feet have reached their stationary landing targets. It preserves every pose, the unloaded swing foot, initial/terminal forces, zero seat force before zero-gap contact, and the existing assumed seat-loading schedule. Its static commands fit the unchanged limits, but the specific command polyline is still too slow. Modeled load allocation is not measured or controller-realized support.</p>
    <img src="force_constraints.png" alt="Late-approach command margins and assumed load allocation">
    <table><thead><tr><th>Check</th><th>Result</th></tr></thead><tbody>{table}</tbody></table>
    <h2>The inherited filter has been checked explicitly</h2><p>The clamp remains at joint limits plus/minus 0.03 rad, the filter coefficient remains 0.1, and the increment cap remains 0.003 rad per 20 ms. The fast polyline check107 clips 88 inverse targets. A separate command-only schedule108 follows the same polyline without target clipping, but takes 25.10 seconds. Independent109 reconstructs its desired path and exactly replays all applied commands in Torch CPU float32. It is not a GPU replay, simulated motion, or proof of minimum execution time.</p>
    <img src="filter_timing.png" alt="All command time checks exceed the existing allowance">
    <table><tr><th>Constructed schedule phase</th><th>Time</th></tr>{phase}</table>
    <p>The next design must jointly account for command headroom, load transfer and transition time. Merely satisfying static joint bounds or slowing this path does not produce an admitted sitting motion. No physical trial or PPO run follows these failed time gates.</p>
    <h2>All recorded attempts in 58.4 seconds</h2><video controls preload="metadata" src="../all_attempts_video_v2_59/all_attempts.mp4"></video><p>The unchanged video covers all 32 recorded sources: 696 complete trials (405 successes and 291 failures), 72 recorded diagnostic prefixes, and explicit static disclosures for 72 aborted case attempts without saved motion. These are historical aggregate attempts, not current-controller accuracy. The independently checked offline schematic uses recorded XYZW/body transforms; 0.1 s pose samples bracket 0.02 s events. No new motion was added in this round.</p>
    <h2>Repair and PPO gates remain unchanged</h2><p>The remaining eight trials are 36 cm/+15° and +20°, 37 cm/+16.25°, and 36.5 cm/+17.5°, two placements each. Housing contacts remain at 36 cm/+20°. All eight must be repaired and all 64 successes retained in matched full 20-second tests before the small two-seed PPO comparison. Fresh holdout stays closed.</p><p>Materials, all-tick slipping, forbidden-body OR illegal-patch contact, stable support, terminal tilt ≤40° and legal seat normal load ≥0.05 body weight are unchanged. There is no new effort result. Verified09 equals current61111 on common full clean pairs; versus selected62111@8 s on 20 old-development common-clean pairs, whole20 torque² is 4.38% higher and final5 is 25.45% higher. No failed-fragment savings are reported.</p>
    <p>Torque is a proxy, not energy. Reference-assisted rigid-slab simulation does not establish unassisted sitting, real-robot performance or sitting and leaning anywhere. Leaning has not broadened.</p>
    <p><a href="summary.json">Detailed summary</a> · <a href="independent_audit105.json">Force witness audit</a> · <a href="independent_audit107.json">Fast filter audit</a> · <a href="independent_audit109.json">Constructed schedule audit</a> · <a href="video_audit.json">Video coverage</a> · <a href="SHA256.json">Hashes</a></p></main></body></html>'''
    (O/'findings.html').write_text(page)
    (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
    print(json.dumps(dict(report=str(O),sha_entries=len(read(O/'SHA256.json')),physical_repair_complete=False)),flush=True)
if __name__=='__main__':main()
