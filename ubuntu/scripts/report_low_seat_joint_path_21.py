"""Export the rejected physical probe and its recorded geometry diagnosis."""
import json,hashlib,shutil,html
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from humanoid_sitting.shared_frame import BodyModel,apply,transform
from plan_seat_load_targets import clipped_vertices
from audit_low_seat_contact_probe_v2 import summarize,clean
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OUT=D/'progress_report_21'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    audit=read(D/'joint_path_17/independent_joint_path_audit_v4_17.json');assert audit['audit_completed'] and audit['input_action_reward_and_planner_checks_passed']
    OUT.mkdir(exist_ok=False);p=D/'joint_path_17/candidate';r=read(p/'results.json');geom=read(p/'geometry.json')
    paired=[]
    for label,base in [('best_normal3s',D/'normal_landing3s_06/candidate'),('original_parent',D/'clearance_damped_01/baseline')]:
        br=read(base/'results.json')['episodes'];cr=r['episodes']
        groups=[('all',list(range(24)))]+[(f"{geom[row['geometry_index']]['height']}/{geom[row['geometry_index']]['pitch_deg']}",[i,i+1]) for i,row in enumerate(cr) if i%2==0]
        for group,cases in groups:
            for strict in [False,True]:
                common=[i for i in cases if clean(br[i]) and clean(cr[i]) and (not strict or (br[i]['traction_bad_ticks']==0 and cr[i]['traction_bad_ticks']==0))];metrics={}
                for window in ['metrics_whole_trial','metrics_last10s','metrics_last5s']:
                    metrics[window]={}
                    if not common:continue
                    for key in br[common[0]][window]:
                        x=np.array([br[i][window][key] for i in common]);y=np.array([cr[i][window][key] for i in common])
                        if x.dtype.kind not in 'fiu' or x.ndim!=1:continue
                        metrics[window][key]=dict(baseline=float(x.mean()),candidate=float(y.mean()),percent_change=float((y.mean()/x.mean()-1)*100) if x.mean()!=0 else None)
                paired.append(dict(baseline=label,group=group,strict_both=strict,n=len(common),common_cases=common,metrics=metrics))
    (OUT/'paired_effort_details.json').write_text(json.dumps(dict(scope='Only common full-duration clean successes in BOTH controllers; strict subsets also require zero conservative margin ticks in BOTH. Empty pairs are reported, never replaced with failed fragments.',rows=paired),indent=2))
    t=np.load(p/'trace.npz');e=np.load(p/'every_tick.npz');plans={x['case']:x for x in read(p/'rolling_plan_audit.json')}
    m=BodyModel('runs/g1_kinematics_geometry_04/model.json');origins=np.asarray(r['origins']);bn=r['body_names'];pelvis=bn.index('pelvis')
    dx=[];curves={}
    for row in r['episodes']:
        i=row['case'];g=geom[row['geometry_index']];a=plans[i]
        first=lambda mask:float(e['t'][np.flatnonzero(mask)[0]]) if np.any(mask) else None
        slip=first(~e['no_slip'][:,i]);forbidden=first(e['wrong'][:,i]>10.);unwanted=first((e['wrong'][:,i]>10.)|(~e['legal'][:,i]))
        duration=7. if 'joint_path_17' in a else 5.;start=a['tick']*.02
        u=np.clip((t['t']-start)/duration*24,0,24);lo=np.minimum(u.astype(int),23);f=u-lo;roots=np.array(a['root_path'])
        target=(1-f[:,None])*roots[lo,:3]+f[:,None]*roots[lo+1,:3];root=t['body'][:,i,pelvis,:3]-origins[i]
        error=root-target;event=min([v for v in [slip,unwanted] if v is not None],default=20.);before=t['t']<=event
        d=dict(case=i,height_m=g['height'],slope_deg=g['pitch_deg'],selected='joint_path_17' in a,stable=row['success'],
            slip_ticks=row['slip_ticks'],illegal_ticks=row['illegal_ticks'],failures=row['failures'],
            first_slip_s=slip,first_forbidden_s=forbidden,first_unwanted_union_s=unwanted,
            fresh_geometry_before_event=bool((t['geometry'][before,i,11]>0).all()),
            measured_normal_error_deg=float(np.rad2deg(np.arccos(np.clip(np.asarray(a['normal'])@np.asarray(g['normal']),-1,1)))),
            measured_offset_error_m=float(a['offset']+np.asarray(g['normal'])@np.asarray(g['top_center'])))
        samples=[]
        for tm in [8,10,12,14,14.6,15,20]:
            k=int(np.argmin(abs(t['t']-tm)));samples.append(dict(t=float(t['t'][k]),root_error_m=error[k].tolist(),
                actual_pitch_rad=float(Rotation.from_quat(t['body'][k,i,pelvis,3:]).as_euler('xyz')[1])))
        d['samples']=samples;dx.append(d)
        if i in [0,6,10]:
            actual=[];nominal=[];SR=Rotation.from_euler('y',np.deg2rad(g['pitch_deg'])).as_matrix();center=np.array(g['top_center'])
            for k in range(len(t['t'])):
                values=[]
                for side in ['left','right']:
                    name=side+'_hip_pitch_link';body=t['body'][k,i,bn.index(name)];T=transform(Rotation.from_quat(body[3:]).as_matrix(),body[:3]-origins[i])
                    local=(apply(T,m.vertices[name])-center)@SR;local[:,0]-=.4;v=clipped_vertices(local)
                    if len(v):values.append(float(v[:,2].min()))
                actual.append(min(values) if values else np.nan)
            qpaths=np.load(p/'last_rolling_paths.npz')['q'][i]
            for qi,rr in zip(qpaths,roots):
                T=transform(Rotation.from_quat(rr[3:]).as_matrix(),rr[:3]);fk=m.fk(qi);values=[]
                for side in ['left','right']:
                    name=side+'_hip_pitch_link';local=(apply(T@fk[name],m.vertices[name])-center)@SR;local[:,0]-=.4;v=clipped_vertices(local)
                    if len(v):values.append(float(v[:,2].min()))
                nominal.append(min(values) if values else np.nan)
            curves[i]=dict(t=t['t'],actual=np.array(actual),nominal=np.interp(t['t'],np.linspace(start,start+duration,25),nominal),event=forbidden,
                normal_error=error@np.asarray(a['normal']),title=f"{g['height']*100:g} cm / +{g['pitch_deg']:g}°")
    (OUT/'recorded_diagnosis_21.json').write_text(json.dumps(dict(scope='Every-tick labels; recorded meshes/poses at0.1s only bracket0.02s events. No single-factor proof. All network/controller geometry was available; live truth is used only by this offline audit.',rows=dx),indent=2))
    fig,ax=plt.subplots(1,2,figsize=(12,4.5));labels=['Stable','Clean','Slip cases','Unwanted contacts'];keys=['stable','clean','slipping','unwanted_contacts'];x=np.arange(4)
    for j,(name,data,color) in enumerate([('Best normal3s',audit['best_probe'],'#237a57'),('Rejected joint path',audit['candidate'],'#b94b43')]):
        bars=ax[0].bar(x+(j-.5)*.34,[data[k] for k in keys],.34,label=name,color=color);ax[0].bar_label(bars,padding=3)
    ax[0].set_xticks(x,labels,rotation=12);ax[0].set_ylim(0,26);ax[0].legend();ax[0].set_title('24 matched development probe episodes')
    for i,c in curves.items():ax[1].plot(c['t'],1000*c['normal_error'],label=c['title'])
    ax[1].set_xlim(8,16);ax[1].axhline(0,color='black',lw=.7);ax[1].set_ylabel('Root tracking error along seat normal (mm)');ax[1].set_xlabel('Episode time (s)');ax[1].legend();ax[1].set_title('Tracking errors near touchdown')
    fig.suptitle('Rejected physical repair — no PPO or fresh holdout',fontsize=14);fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),sharey=True)
    for ax,(i,c) in zip(axes,curves.items()):
        ax.plot(c['t'],1000*c['nominal'],label='Planned housing mesh');ax.plot(c['t'],1000*c['actual'],label='Recorded housing mesh');ax.axhline(0,color='black',lw=.7)
        ax.axvline(c['event'],color='red',ls='--',label='First >10 N forbidden contact');ax.set_xlim(13.5,15.5);ax.set_ylim(-7,30);ax.set_title(c['title']);ax.set_xlabel('Time (s)')
    axes[0].set_ylabel('Mesh clearance above true seat plane (mm)');axes[-1].legend(fontsize=7);fig.suptitle('Offline mesh diagnosis: 0.1s samples bracket contact events');fig.tight_layout();fig.savefig(OUT/'housing_diagnosis.png',dpi=160);plt.close(fig)
    copies={
        'physical_audit.json':D/'joint_path_17/independent_joint_path_audit_v4_17.json','sensor_control_audit.json':p/'sensor_control_audit.json',
        'physical_revision.json':D/'joint_path_17/revision.json','candidate_results.json':p/'results.json',
        'model_preparation_17.json':D/'joint_path_preparation_17.json','earlier_approach_screen_15.json':D/'earlier_approach_screen_15.json','joint_path_screen_16.json':D/'joint_path_screen_16.json',
        'clearance_endpoint_screen_18.json':D/'clearance_endpoint_screen_18.json','clearance_endpoint_buffer_screen_19.json':D/'clearance_endpoint_buffer_screen_19.json','clearance_velocity_screen_20.json':D/'clearance_velocity_screen_20.json',
        'audit_recovery.json':D/'joint_path_audit_recovery_17/recovery.json','prior_hash_recheck.json':D/'prior_recheck_during_joint_path_17.json',
        'unselected_trace_differences.json':D/'joint_path_unselected_difference_17.json'}
    for dest,source in copies.items():shutil.copy2(source,OUT/dest)
    rows=''.join(f"<tr><td>{d['height_m']*100:g} cm / +{d['slope_deg']:g}°</td><td>{d['case']}</td><td>{d['stable']}</td><td>{d['slip_ticks']}</td><td>{d['first_forbidden_s']}</td></tr>" for d in dx if d['selected'])
    text=f'''<!doctype html><meta charset="utf-8"><title>Low-seat repair: rejected joint path</title><style>body{{font:17px/1.5 system-ui;max-width:1150px;margin:36px auto;padding:0 24px;background:#f6f7f8;color:#18262c}}img{{width:100%;background:white}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #ddd}}.notice{{padding:20px;background:#fff0ed;border-left:5px solid #b94b43}}a{{color:#165d82}}</style>
    <h1>Low-seat repair remains unresolved</h1><div class="notice">The new joint-space approach is rejected: <b>16/24 stable,14/24 clean,6 slipping,8 housing-contact cases</b>. Best normal3s had22 stable,14 clean,10 slipping and2 contacts. The six targeted trials added forbidden hip-housing contacts; four lost their slip flags but still failed contact safety. There were no falls,impact or joint-limit failures. No established clean probe case was lost. No PPO or fresh holdout was run.</div>
    <p>The best independently verified full-development controller remains guarded normal3s with frozen early61111 and resting59117: <b>64/72 clean,all60 original successes retained</b>,eight slipping cases andtwo housing contacts. This new isolated-parent probe does not replace that72-case result.</p>
    <img src="comparison.png" alt="Probe outcomes and tracking errors"><h2>What was tested</h2><p>Only measured height&lt;38cm and14&lt;positive slope&lt;18° received the8–15s path. The original first10s lock explicitly became8s for these six trials. The exact-preservation audit also FAILED: unselected measured geometry first diverged at8.2s, before their commands diverged after10s. Maximum unselected body-pose component difference was2.08mm and joint-position difference0.00265rad. The cause of this coupling remains unresolved; no tolerance was relaxed. The deepest36cm/+20° case kept its previous controller routing. All14 fingers,materials,physical criteria and applied0.15rad/s slew were preserved. The frozen resting policy used current sensor inputs;there was no optimizer.</p>
    <h2>Recorded contact diagnosis</h2><p>First new forbidden housing contacts occurred at14.60–14.70s. Around14.6s,root position was12–15mm farther inward and3.4–4.6mm below its planned path. Nominal model clearance did not survive the physical tracking error. Sensor geometry was available before all events. This is consistent with tracking and rate limitations,not a single-factor causal proof. Recorded mesh poses are sampled every0.1s and only bracket0.02s contact labels.</p><img src="housing_diagnosis.png" alt="Planned and actual housing clearances">
    <table><tr><th>Condition</th><th>Case</th><th>Stable</th><th>Slip ticks</th><th>First forbidden contact(s)</th></tr>{rows}</table>
    <h2>Audits and effort</h2><p>The completed independent audit verified exact starts,first8s,actual417 sensor/action-memory channels,frozen parent,full24000 active ticks,600 initial settling ticks,zero optimizer updates,fixed fingers,bounded corrections and sampled applied slew. Offline replay was corrected to use original float32 inputs and the exact frozen reference inertial model instead of newly recorded COM metadata. The2e−7 tolerance is unchanged;both earlier audit failures are preserved. All24,242 prior-study hashes remain intact.</p>
    <p>All14 best-probe clean episodes remained clean, but their traces were not identical; paired effort differences are reported without an overall improvement claim. Reduced slip counts in newly failed housing-contact episodes are not counted as success or effort savings. Whole20s,last10s,last5s torque²,absolute,RMS,leg/upper metrics and strict subsets are in <a href="paired_effort_details.json">the paired details</a>. Conservative warnings remain separate from physical failures. The earlier verified full-development comparison still has mixed effort results;this is not a new PPO gain.</p>
    <p>Deepest-seat screens18–20 remain unadmitted: endpoint-only clearance was insufficient along the path;adding a buffer pushed nominal rate above the unchanged cap. Matching recorded incoming command velocity reduced that excess but still failed (0.150119rad/s). No tolerance was relaxed and no physical trial used those designs.</p>
    <p>These are reference-assisted simulation experiments on rigid slabs. They establish neither arbitrary-surface sitting/leaning nor real-robot validation. Torque is an effort proxy,not energy. <a href="recorded_diagnosis_21.json">Recorded diagnosis</a> · <a href="SHA256.json">File hashes</a></p>'''
    (OUT/'findings.html').write_text(text)
    (OUT/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in OUT.iterdir() if p.is_file()},indent=2))
    print(str(OUT))
if __name__=='__main__':main()
