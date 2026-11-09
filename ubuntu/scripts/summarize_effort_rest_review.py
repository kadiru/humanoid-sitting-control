"""Read-only supplemental diagnosis and reporting for a completed frozen study."""
import argparse, hashlib, html, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--review',required=True)
a=p.parse_args();root=Path(a.study);out=Path(a.review)
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
protocol=read(root/'protocol.json');report=read(out/'results.json');rows=report['details']
audit={'snapshots':{},'run_manifests':{},'command_samples':{},'initial_actor_outputs_zero':True}
for i,(name,digest) in enumerate(protocol['hashes'].items()):
    path=root/'frozen'/f'{i:03d}_{Path(name).name}'
    assert sha(path)==digest,path
    audit['snapshots'][str(path)]=digest
for directory in sorted(root.glob('train_*'))+sorted(root.glob('eval_*')):
    if not directory.is_dir():continue
    manifest=read(directory/'manifest.json')
    for i,(name,digest) in enumerate(manifest['source_hashes'].items()):
        assert protocol['hashes'][str(Path(name).resolve())]==digest
        assert sha(directory/'source'/f'{i:02d}_{Path(name).name}')==digest
    for name,digest in manifest['bank_hashes'].items():
        assert protocol['hashes'][str(Path(manifest['args']['bank'])/name)]==digest
    audit['run_manifests'][directory.name]={'saved_sources_match':True,'bank_matches':True}

metrics=['total_squared_torque_Nm2','total_absolute_torque_Nm','all_joint_rms_Nm',
         'normalized_torque_squared','leg_squared_torque_Nm2','upper_squared_torque_Nm2',
         'support_load_bw','foot_load_bw','joint_speed_rms']
def group(items):
    common=[x for x in items if x['common_success']]
    result={'success':sum(x['success'] for x in items),'total':len(items),'common_successes':len(common),
            'physical_failures':{k:sum(x['failures'][k] for x in items) for k in items[0]['failures']},
            'saving_cases':[f"{x['bank']}/{x['case']}" for x in common if x['late_effort_change_percent']<0],
            'metrics':{}}
    for window in ['metrics_whole_trial','metrics_last10s']:
        result['metrics'][window]={}
        for metric in metrics:
            fixed=np.array([x['fixed'][window][metric] for x in common]);ppo=np.array([x['ppo'][window][metric] for x in common])
            result['metrics'][window][metric]={'fixed':float(fixed.mean()) if len(common) else None,
                'ppo':float(ppo.mean()) if len(common) else None,
                'pooled_change_percent':float(100*(ppo.mean()/fixed.mean()-1)) if len(common) and fixed.mean()!=0 else None}
    return result
groups={}
for seed in protocol['seeds']:
    items=[x for x in rows if x['seed']==seed];groups[str(seed)]={'all':group(items)}
    for dim,getter in [('kind',lambda x:x['geometry']['kind']),('bank',lambda x:x['bank']),
                       ('height_cm',lambda x:round(x['geometry']['height']*100,4)),
                       ('slope_deg',lambda x:x['geometry']['pitch_deg']),
                       ('outline',lambda x:x['geometry'].get('shape','rect'))]:
        groups[str(seed)][dim]={str(key):group([x for x in items if getter(x)==key]) for key in sorted(set(getter(x) for x in items))}

sensor=[];series={};diagnostics=[]
for bank in ['train','holdout']:
    fixed=Path(protocol['fixed_baselines'][bank])
    for policy,directory in [('fixed',fixed)]+[(str(seed),root/f'eval_{seed}_{bank}') for seed in protocol['seeds']]:
        result=read(directory/'results.json');geometry=read(directory/'geometry.json');tr=np.load(directory/'trace.npz')
        startup=np.load(directory/'startup.npz');first={x['case']:x['t'] for x in read(directory/'first_failures.json')}
        pelvis=result['body_names'].index('pelvis');cmd=tr['command']-startup['command'][None]
        audit['command_samples'][policy+'_'+bank]={'max_absolute_offset_deg':float(np.rad2deg(abs(cmd).max())),
                                                 'max_100ms_delta_rad':float(abs(np.diff(tr['command'],axis=0)).max())}
        for i,g in enumerate(geometry):
            local=tr['body'][:,i,pelvis,:3]-np.array(result['origins'][i]);rot=Rotation.from_quat(tr['body'][:,i,pelvis,3:]).as_matrix()
            normal=np.einsum('nji,j->ni',rot,np.array(g['normal']));offset=(local-g['top_center'])@g['normal']
            z=tr['geometry'][:,i];valid=z[:,11].astype(bool)
            offset_error=abs(z[:,5]-offset)*1000
            normal_error=np.rad2deg(np.arccos(np.clip((z[:,2:5]*normal).sum(1),-1,1)))
            before=tr['t']<first.get(i,float('inf'))-1.e-10
            first_non_good=float(tr['t'][np.flatnonzero(~tr['good'][:,i])[0]]) if (~tr['good'][:,i]).any() else None
            for scope,mask in [('before_physical_failure',before),('qualifying_motion',before&tr['good'][:,i])]:
                chosen=mask&valid
                sensor.append({'policy':policy,'bank':bank,'case':i,'kind':g['kind'],'slope_deg':g['pitch_deg'],'scope':scope,
                    'samples':int(mask.sum()),'valid':int(chosen.sum()),'lost':int((mask&~valid).sum()),
                    'p95_offset_mm':float(np.percentile(offset_error[chosen],95)) if chosen.any() else None,
                    'max_offset_mm':float(offset_error[chosen].max()) if chosen.any() else None,
                    'p95_normal_deg':float(np.percentile(normal_error[chosen],95)) if chosen.any() else None,
                    'first_nonqualifying_sample_s':first_non_good})
            series[(policy,bank,i)]={'t':tr['t'].copy(),'good':tr['good'][:,i].copy(),'support':tr['support'][:,i].copy(),
                                    'effort':tr['effort'][:,i].copy(),'offset_error':offset_error,'normal_error':normal_error,'valid':valid}
            if policy!='fixed':
                e=result['episodes'][i];late=tr['t']>10;mean_offset=np.rad2deg(cmd[late,i].mean(0));largest=np.argsort(abs(mean_offset))[-5:][::-1]
                diagnostics.append({'seed':int(policy),'bank':bank,'case':i,'success':e['success'],
                    'first_nonqualifying_10Hz_sample_s':first_non_good,
                    'late_sample_fraction_under_minimum_support':float((tr['support'][late,i]<.05).mean()),
                    'largest_final10s_command_offsets_deg':{result['joint_names'][j]:float(mean_offset[j]) for j in largest},
                    'full_window_good_fraction':e['good_fraction'],'final_hold_s':e['final_hold_s']})

supplement={'groups':groups,'sensor_by_case':sensor,'motion_diagnostics':diagnostics,'additional_validation':audit,
 'sensor_scope':'10Hz saved samples, includes comparison to fixed. Qualifying motion is the saved good mask, not only successful trials. No inference of 50Hz accuracy.',
 'failure_scope':'Physical labels are latched. Missing success without a physical flag means the unchanged support/traction/slip/motion qualification failed. Individual failed gate channels were not stored; do not infer them from good alone.',
 'initialization_scope':'Distinct seeds intentionally have different hidden weights and critics; both actor output heads are exactly zero, verified in initial_policy_check and primary verifier. Baseline has no trainable actor.',
 'command_scope':'Runtime asserts bound every50Hz tick. Offline slew and constant-command checks use saved10Hz commands; initial observations/commands match exactly.',
 'finding':'No overall effort saving or stability retention. Most additional effort is in legs. Isolated savings cannot justify policy promotion.'}
(out/'supplement.json').write_text(json.dumps(supplement,indent=2))
fig,axes=plt.subplots(2,3,figsize=(14,7),sharex=True)
examples=[('train',1,'30 cm sitting: more leg effort'),('train',3,'40 cm sitting: isolated saving'),('train',8,'60° leaning: support unloads')]
for col,(bank,i,title) in enumerate(examples):
    for policy,color in [('fixed','#6b7973'),('54511','#247e98'),('54517','#d59140')]:
        s=series[(policy,bank,i)]
        axes[0,col].plot(s['t'],s['effort'],color=color,label=policy)
        axes[1,col].plot(s['t'],s['support'],color=color,label=policy)
    axes[0,col].set_title(title);axes[1,col].set_xlabel('Seconds from supported start')
    axes[1,col].axhline(.05,color='#ba4a4a',ls=':',lw=1,label='Minimum support load')
    for row in range(2):axes[row,col].spines[['top','right']].set_visible(False)
axes[0,0].set_ylabel('Total squared torque (Nm²)');axes[1,0].set_ylabel('Surface normal load / body weight')
axes[0,0].legend();axes[1,2].legend();fig.tight_layout();fig.savefig(out/'motion_diagnosis.png',dpi=180);plt.close(fig)

def fmt(v):return '—' if v is None else f'{v:.2f}'
table=''
for seed in protocol['seeds']:
    g=groups[str(seed)]['all']
    for window in ['metrics_whole_trial','metrics_last10s']:
        for metric in metrics:
            m=g['metrics'][window][metric]
            table+=f'<tr><td>{seed}</td><td>{"0–20 s" if window=="metrics_whole_trial" else "10–20 s"}</td><td>{metric}</td><td>{g["common_successes"]}</td><td>{fmt(m["fixed"])}</td><td>{fmt(m["ppo"])}</td><td>{fmt(m["pooled_change_percent"])}%</td></tr>'
sensor_table=''
for x in sensor:
    if x['bank']=='holdout' and x['case']>=13 and x['scope']=='before_physical_failure':
        sensor_table+=f'<tr><td>{x["policy"]}</td><td>{x["case"]}</td><td>{x["valid"]}/{x["samples"]}</td><td>{fmt(x["p95_offset_mm"])}</td><td>{fmt(x["p95_normal_deg"])}</td></tr>'
intro='''<div style="background:#fff1df;border-left:5px solid #ae5919;padding:16px"><strong>Result: PPO did not achieve lower-effort supported rest.</strong><p>Fixed posture passed 29/29. Seeds 54511 and 54517 passed 11/29 and 15/29. All successes were sitting; leaning was 0/12 in each seed versus 12/12 fixed. No falls, impact, torque-limit or joint-limit failures occurred. Each policy triggered forbidden contact on held-out sitting case1 (30°), at0.06s /0.14s. Other unsuccessful trials lost the support qualification or spent too long outside it; these are not reported as falls.</p><p>Even on common full-duration successes, mean total squared torque increased. Most increase came from the legs; upper-body effort changed little. The 40cm training seat improved in both seeds (−54.2% /−50.6% final10s); seed54517 also improved on one held-out10° seat (−13.6%). These isolated improvements do not establish a usable general policy. No policy is promoted.</p></div>'''
append='''<h2>What the saved motion shows</h2><p>At30cm the policy reduces surface support and increases motor effort. At40cm it increases support and reduces effort. On45°/60° leaning, both policies reduce surface load substantially and lose qualification, although the measured geometry remains accurate and available. This supports a learned posture/load-transfer problem, not a universal perception explanation. No simulation or additional training was run for this report.</p><img src="motion_diagnosis.png"><p>The reward is correctly decreasing in total squared torque when support qualifies. Outside that qualification it provides zero positive credit; a separate penalty applies to the first physical failure. The saved records do not establish why PPO converged to the worse policy. A flat reward outside valid support is a plausible optimization difficulty, not a proven causal diagnosis. Friction, slip and motion gates were evaluated but not individually recorded in these traces, so their relative contribution cannot be reconstructed exactly.</p><h2>Matched effort, including both time windows</h2><p>Each seed uses its own common successful cases (11 /15); these subsets differ. Metrics are averages of per-trial50Hz means. Changes are ratios of pooled means, not the mean of per-case percentages. Failed trial fragments are excluded. Nm RMS means the time average of instantaneous all37-joint RMS. Nm² and absolute torque are effort proxies, not energy.</p><table><tr><th>Seed</th><th>Window</th><th>Metric</th><th>Pairs</th><th>Fixed</th><th>PPO</th><th>Change</th></tr>'''+table+'''</table><h2>Tracking caveat on steep held-out surfaces</h2><p>All training-surface saved samples retained geometry with p95 offset error below1.7mm and normal error below0.29°. The75° held-out surfaces have missing observations and sometimes an approximately90° wrong plane even when marked available. Fixed posture has related errors too; it ignores these measurements. The table is the10Hz pre-physical-failure sample audit, not a50Hz guarantee. supplement.json also separates qualifying-motion samples and every surface. Accurate training-surface tracking does not rescue either leaning policy, so this issue cannot explain all failures.</p><table><tr><th>Policy</th><th>Held-out case</th><th>Available samples</th><th>p95 offset mm</th><th>p95 normal °</th></tr>'''+sensor_table+'''</table><h2>Additional audit</h2><p>All105 frozen files and61 prior-file hashes match. Every frozen snapshot and all6 run source snapshots / bank manifests match. Both256-update budgets completed: seed54511=409845 active+16139 reset-warmup, seed54517=408961+17023. Total851968, including33162 warmup; the separate3328-transition smoke and global settle are excluded. Actual final checkpoint weights, exact startup arrays,1000-step horizons, fixed/uncontrolled commands and learned command bounds passed. Actor and critic changed and remain finite. Two seeds intentionally have different hidden initialization but the same zero-output starting behavior. The robot starts supported, and this does not demonstrate a standing transition.</p><p><a href="supplement.json">Grouped results by seed, kind, height, slope and outline; per-case tracking; motion and snapshot audits</a></p>'''
original=(out/'review.html').read_text()
assert not (out/'review_initial.html').exists()
(out/'review_initial.html').write_text(original)
original=original.replace('<img src="stability.png">',intro+'<img src="stability.png">',1)
(out/'review.html').write_text(original+append)
print(json.dumps({seed:value['all'] for seed,value in groups.items()},indent=2))
print('STEEP SENSOR',json.dumps([x for x in sensor if x['bank']=='holdout' and x['case']>=13 and x['scope']=='before_physical_failure']))
