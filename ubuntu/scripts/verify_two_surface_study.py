"""Read-only, independent audit and report for the frozen two-surface study.

No simulator imports, training, edits to results, or additional evaluations.
--partial audits only completed blocks while the persistent study is running.
"""
import argparse
import csv
import hashlib
import html
import json
import shutil
from pathlib import Path

import numpy as np
import torch


def read(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, indent=2, allow_nan=False))


def equal_tree(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(equal_tree(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal_tree(x, y) for x, y in zip(a, b))
    return a == b


def load(p):
    return torch.load(p, map_location='cpu', weights_only=False)


def finite_changed(initial, final):
    changes = {}
    for role in ['actor_state_dict', 'critic_state_dict']:
        assert all(torch.isfinite(v).all() for v in final[role].values()), role
        changes[role] = [k for k, v in final[role].items() if not torch.equal(v, initial[role][k])]
        assert any(k.startswith('mlp') for k in changes[role]), role
    return changes


def manifest_check(folder, protocol):
    m = read(folder / 'manifest.json')
    assert m['observations'] == 405 and m['actions'] == 23
    for i, (path, digest) in enumerate(m['sources'].items()):
        assert protocol['hashes'][path] == digest, path
        assert sha(folder / 'source' / f'{i:02d}_{Path(path).name}') == digest, path
    for name, digest in m['bank_hashes'].items():
        assert sha(Path(m['args']['bank']) / name) == digest, name
    return m


def evaluation_check(folder, protocol, checkpoint=None, expected_count=2):
    m = manifest_check(folder, protocol)
    d = read(folder / 'results.json')
    rows = d['episodes']
    assert len(rows) == d['count'] == expected_count
    assert d['successes'] == sum(x['success'] for x in rows)
    assert len({(x['surface_id'], x['start_stage']) for x in rows}) == expected_count
    for row in rows:
        assert row['steps'] == 1000 and row['duration_s'] == 20
        assert row['late_ticks'] == 500 and row['last5_ticks'] == 250
        assert row['success'] == (not any(row['failures'].values()) and row['final_hold_s'] >= 5)
        for window in ['metrics_whole_trial', 'metrics_last10s', 'metrics_last5s']:
            assert all(np.isfinite(v) for v in row[window].values())
            z = row[window]
            assert abs(z['total_squared_torque_Nm2'] - z['leg_squared_torque_Nm2'] - z['upper_squared_torque_Nm2']) < max(.01, z['total_squared_torque_Nm2'] * 1e-5)
    if checkpoint:
        assert m['args']['checkpoint'] == str(checkpoint)
        actual, expected = load(folder / 'initial.pt'), load(checkpoint)
        for role in ['actor_state_dict', 'critic_state_dict']:
            assert equal_tree(actual[role], expected[role]), f'{folder.name}: wrong {role}'
        assert actual['iter'] == expected['iter']
    else:
        assert m['args']['checkpoint'] is None
    trace = np.load(folder / 'trace.npz')
    start = np.load(folder / 'startup.npz')
    assert trace['command'].shape == (200, expected_count, 37)
    assert np.allclose(trace['t'], np.arange(1, 201) / 10, rtol=0, atol=1e-12)
    assert all(np.isfinite(trace[k]).all() for k in trace.files)
    # The 23 actuated body-joint names are also present in the frozen deployment module.
    # The remaining 14 finger joints never receive policy offsets.
    assert all('joint' in n for n in d['joint_names']) and len(d['joint_names']) == 37
    ids = [j for j, n in enumerate(d['joint_names']) if not any(x in n for x in ['_zero_', '_one_', '_two_', '_three_', '_four_', '_five_', '_six_'])]
    other = [j for j in range(37) if j not in ids]
    assert len(ids) == 23 and len(other) == 14
    assert np.array_equal(trace['command'][:, :, other], np.broadcast_to(start['command'][None, :, other], trace['command'][:, :, other].shape))
    bank = np.load(Path(m['args']['bank']) / 'reset_candidates.npz')['command']
    span = np.array(m['spans'])
    bound_error = 0.
    for row in rows:
        c, surface = row['case'], row['surface_id']
        low = bank[surface, :, ids].min(axis=1) - span
        high = bank[surface, :, ids].max(axis=1) + span
        # Filtered commands can lag the instantaneous reference, but stay in
        # the convex envelope of the entire path plus the bounded residual.
        commands = trace['command'][:, c][:, ids]
        err = max(float((low - commands).max()), float((commands - high).max()), 0.)
        assert err <= 2e-6, (folder.name, c, err)
        bound_error = max(bound_error, err)
    sequence = np.concatenate((start['command'][None], trace['command']), axis=0)
    max_100ms_change = float(np.abs(np.diff(sequence[:, :, ids], axis=0)).max())
    assert max_100ms_change <= .010002, max_100ms_change
    fixed = m['args']['mode'] == 'fixed'
    if fixed:
        assert np.array_equal(trace['command'], np.broadcast_to(start['command'][None], trace['command'].shape))
    reference_replay_error=None
    if m['args']['mode']=='reference':
        commands=start['command'].copy(); maximum=0.
        for tick in range(1000):
            for row in rows:
                c=row['case']; phase=min(6.,row['start_stage']+tick*.01)
                lo=int(np.floor(phase));hi=min(lo+1,6);a=phase-lo;a=a*a*(3-2*a)
                ref=bank[row['surface_id'],lo]*(1-a)+bank[row['surface_id'],hi]*a
                commands[c,ids]+=np.clip(.1*(ref[ids]-commands[c,ids]),-.002,.002)
            if tick%5==4:
                maximum=max(maximum,float(np.abs(commands-trace['command'][tick//5]).max()))
        # Validated fixed commands at every knot are unchanged by joint-limit
        # clipping. The interpolated path stays inside their convex envelope.
        assert maximum<3e-6,maximum
        reference_replay_error=maximum
    # Audit the supplied offline reference without importing its runtime implementation.
    for row in rows:
        c = row['case']
        for tick, t in enumerate(trace['t']):
            phase = row['start_stage'] if fixed else min(6., row['start_stage'] + (t - .02) / 2.)
            lo = int(np.floor(phase)); hi = min(lo + 1, 6); a = phase - lo; a = a*a*(3-2*a)
            ref = bank[row['surface_id'], lo] * (1-a) + bank[row['surface_id'], hi] * a
            assert np.max(np.abs(ref - trace['reference'][tick, c])) < 2e-6
    first = read(folder / 'first_failures.json')
    assert {r['case'] for r in first} == {r['case'] for r in rows if any(r['failures'].values())}
    sensor = read(folder / 'sensor_audit.json')
    assert sensor['channels'] == 405 and sensor['physics_advances_during_camera'] == 0
    cutoff = {r['case']: r['t'] for r in first}
    geometry = []
    for row in rows:
        c = row['case']
        before = [a for a in sensor['truth_validation'] if a['case'] == c and a['t'] < cutoff.get(c, 20.01)]
        valid = [a for a in before if a['valid']]
        mask = trace['t'] < cutoff.get(c, 20.01)
        motion = mask & ((trace['root_speed'][:, c] > .02) | (trace['gyro'][:, c] > .05))
        geometry.append(dict(case=c, surface_id=row['surface_id'], stage=row['start_stage'], samples_before_failure=len(before), valid_samples=len(valid), available_fraction=len(valid)/len(before) if before else None, max_abs_offset_error_mm=max([abs(a['offset_error_m'])*1000 for a in valid], default=None), max_normal_error_deg=max([a['normal_error_deg'] for a in valid], default=None), motion_samples_10hz=int(motion.sum()), motion_missing_samples_10hz=int((trace['geometry'][motion,c,11] == 0).sum())))
    trace_effort = (trace['torque'].astype(np.float64)**2).sum(-1)
    assert np.allclose(trace_effort, trace['effort'], rtol=2e-6, atol=.02)
    diagnostics=[]
    for row in rows:
        c=row['case']; late=trace['t']>15
        residual_fraction=np.abs((trace['command'][late,c][:,ids]-trace['reference'][late,c][:,ids])/span)
        diagnostics.append(dict(case=c,surface_id=row['surface_id'],stage=row['start_stage'],success=row['success'],physical_failures=row['failures'],first_failure=next((x for x in first if x['case']==c),None),final_hold_s=row['final_hold_s'],sampled_last5s=dict(good_fraction=float(trace['good'][late,c].mean()),legal_fraction=float(trace['legal'][late,c].mean()),traction_fraction=float(trace['traction'][late,c].mean()),no_slip_fraction=float(trace['no_slip'][late,c].mean()),support_load_bw_mean=float(trace['support'][late,c].mean()),support_load_bw_min=float(trace['support'][late,c].min()),max_root_speed_m_s=float(trace['root_speed'][late,c].max()),max_gyro_rad_s=float(trace['gyro'][late,c].max()),max_tilt_deg=float(np.rad2deg(trace['tilt'][late,c]).max()),maximum_command_residual_fraction_of_span=float(residual_fraction.max()))))
    return dict(count=len(rows), successes=d['successes'], physical_failures=sum(any(r['failures'].values()) for r in rows), by_flag={k: sum(r['failures'][k] for r in rows) for k in rows[0]['failures']}, standing_successes=sum(r['success'] for r in rows if r['start_stage']==0), constant_fixed_commands=fixed, reference_command_replay_max_error_rad=reference_replay_error, uncontrolled_commands_constant=True, command_envelope_error_rad=bound_error, sampled_max_100ms_command_change_rad=max_100ms_change, geometry=geometry,diagnostics=diagnostics), d


def comparisons(label, policy, name, baseline):
    paired = []
    bycase = {(r['surface_id'],r['start_stage']):r for r in baseline['episodes']}
    for p in policy['episodes']:
        b = bycase[(p['surface_id'],p['start_stage'])]
        if not (p['success'] and b['success']):
            continue
        for window in ['metrics_whole_trial','metrics_last10s','metrics_last5s']:
            for metric in p[window]:
                pv, bv = p[window][metric], b[window][metric]
                paired.append(dict(policy=label, baseline=name, surface_id=p['surface_id'], start_stage=p['start_stage'], window=window, metric=metric, policy_value=pv, baseline_value=bv, change_percent=100*(pv/bv-1) if bv else None))
    return paired


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--partial',action='store_true')
    args = parser.parse_args(); study=args.study; out=args.output
    out.mkdir(exist_ok=False)
    shutil.copy2(Path(__file__),out/'verification_source.py')
    protocol=read(study/'protocol.json')
    for i,(path,digest) in enumerate(protocol['hashes'].items()):
        assert sha(path)==digest,path
        assert sha(study/'frozen'/f'{i:03d}_{Path(path).name}')==digest,path
    for path,digest in protocol['prior_hashes'].items():
        assert sha(path)==digest,path
    report=dict(passed=False,partial=args.partial,verifier_sha256=sha(__file__),frozen_hashes_checked=len(protocol['hashes']),prior_hashes_checked=len(protocol['prior_hashes']),training={},gates={},evaluations={},pairwise_effort=[],limitations=['Two known rigid surfaces and 14 previously used starts; no unseen-surface or disturbance test.','Two seeds share the same conditions; repetitions are not independent generalization evidence.','This is residual PPO around an offline reference motion. Actor and critic also receive the reference and phase.','Whole-body torque is an effort proxy, not electrical energy or a global optimum.','Effort comparisons include only common full-duration support successes. Fixed standing is a retention baseline, not a successful sitting trajectory.','Geometry truth is used only for reset construction, reward and offline labels; reference paths are an explicit offline scaffold.','Command traces are sampled at 10 Hz; the frozen controller applies its slew constraint at 50 Hz.'])
    all_total=0
    for seed in protocol['seeds']:
        state=read(study/f'worker_{seed}.json'); history=state['history']
        previous=None; total=0; active=0; warmup=0; updates=0; audits=[]
        for i,h in enumerate(history):
            expected_stage=6 if not i else history[i-1]['stage']-int(history[i-1]['passed'])
            expected_block=1 if not i or history[i-1]['passed'] else 2
            assert h['stage']==expected_stage and h['block']==expected_block
            if i:assert history[i-1]['passed'] or history[i-1]['block']==1
            folder=study/h['train'];m=manifest_check(folder,protocol);v=read(folder/'verification.json');p=read(folder/'progress.json')
            assert v==h['budget'] and all(p[k]==v[k] for k in v)
            assert v['updates']==32 and v['transitions']==32768 and v['active']+v['warmup']==32768
            assert m['args']['start_stage']==str(h['stage']) and m['args']['seed']==seed
            initial,final=load(folder/'initial.pt'),load(folder/'final.pt')
            changes=finite_changed(initial,final)
            if previous:
                before=load(previous)
                for key in ['actor_state_dict','critic_state_dict','optimizer_state_dict']:
                    assert equal_tree(initial[key],before[key]),(folder.name,key)
                assert initial['iter']==before['iter']+1
                assert m['args']['checkpoint']==str(previous)
            else:
                assert m['args']['checkpoint'] is None and initial['iter']==0
                assert torch.count_nonzero(initial['actor_state_dict']['mlp.4.weight'])==0
                assert torch.count_nonzero(initial['actor_state_dict']['mlp.4.bias'])==0
            updates+=v['updates'];total+=v['transitions'];active+=v['active'];warmup+=v['warmup']
            assert final['iter']==updates-1,(folder.name,final['iter'],updates)
            eps=read(folder/'results.json')['episodes']
            pending_warm=len(eps)*25-v['warmup']; unfinished=v['active']-sum(x['steps'] for x in eps)
            assert 0<=pending_warm<=16*25 and 0<=unfinished<=16*1000
            checkpoint=folder/'final.pt';assert str(checkpoint)==h['checkpoint']
            gate,_=evaluation_check(study/h['evaluation'],protocol,checkpoint)
            assert h['successes']==gate['successes'] and h['passed']==(gate['successes']==2)
            report['gates'][h['evaluation']]=gate
            audits.append(dict(stage=h['stage'],block=h['block'],checkpoint_sha256=sha(checkpoint),budget=v,final_iteration=final['iter'],finite_changed_weights=changes,pending_warmup_transitions=pending_warm,unfinished_active_transitions=unfinished,gate_successes=h['successes']))
            previous=checkpoint
        assert total<=protocol['max_transitions_per_seed'] and updates<=protocol['max_updates_per_seed']
        if not args.partial:
            assert state['phase']=='complete'
            assert state['updates']==updates and state['transitions']==total
            assert state['active_transitions']==active and state['warmup_transitions']==warmup
            assert state['last_checkpoint']==str(previous)
            passed=[h['checkpoint'] for h in history if h['passed']]
            assert state['last_passing_checkpoint']==(passed[-1] if passed else None)
            assert (history[-1]['stage']==0 and history[-1]['passed']) or (history[-1]['block']==2 and not history[-1]['passed'])
        report['training'][str(seed)]=dict(state=state,completed_updates=updates,completed_transitions=total,active=active,warmup=warmup,blocks=audits)
        all_total+=total
    assert all_total<=protocol['max_total_transitions']
    report['total_completed_transitions']=all_total
    if args.partial:
        report['passed']=True;save(out/'validation.json',report);print(json.dumps(dict(passed=True,partial=True,completed_transitions=all_total)));return
    assert read(study.parent/'status.json')['phase'] in ['complete_pending_verification','verified_complete']
    results={}
    for name in ['fixed_all','reference_all']:
        report['evaluations'][name],results[name]=evaluation_check(study/name,protocol,expected_count=14)
    reference_start=np.load(study/'fixed_all/startup.npz')
    expected_names=['reference_all']
    for seed in protocol['seeds']:
        state=read(study/f'worker_{seed}.json')
        variants={'last':state['last_checkpoint']}
        if state['last_passing_checkpoint'] and state['last_passing_checkpoint']!=state['last_checkpoint']:
            variants['last_passing']=state['last_passing_checkpoint']
        for variant,checkpoint in variants.items():
            name=f'eval_{seed}_{variant}';expected_names.append(name)
            report['evaluations'][name],results[name]=evaluation_check(study/name,protocol,Path(checkpoint),14)
            history=state['history'];end=next(i for i,h in enumerate(history) if h['checkpoint']==checkpoint)
            report['evaluations'][name]['trained_start_stages']=sorted({h['stage'] for h in history[:end+1]})
            gate_name=history[end]['evaluation'];gate_folder=study/gate_name
            gs=np.load(gate_folder/'startup.npz');fs=np.load(study/name/'startup.npz')
            stage=history[end]['stage'];indices=[stage,7+stage]
            differences={k:float(np.abs(gs[k]-fs[k][indices]).max()) for k in gs.files}
            differences['root_translation_m']=float(np.abs(gs['root'][:,:3]-fs['root'][indices,:3]).max())
            differences['root_quaternion_component']=float(np.abs(gs['root'][:,3:]-fs['root'][indices,3:]).max())
            gate_rows=read(gate_folder/'results.json')['episodes']
            pair=[dict(surface_id=s,stage=stage,gate_success=gate_rows[s]['success'],final_success=results[name]['episodes'][indices[s]]['success']) for s in [0,1]]
            report['evaluations'][name]['gate_final_comparison']=dict(gate=gate_name,startup_max_differences=differences,outcomes=pair,context='Gate: 2 environments on training GPU. Final: 14 environments on GPU0. These cross-layout starts are not claimed exact; final controller-to-baseline starts must be exact.')
            for baseline in ['fixed_all','reference_all']:
                report['pairwise_effort']+=comparisons(name,results[name],baseline,results[baseline])
    startup_differences={}
    fixed_geometry=read(study/'fixed_all/geometry.json')
    for name in expected_names:
        current=np.load(study/name/'startup.npz')
        errors={k:float(np.max(np.abs(current[k]-reference_start[k]))) for k in reference_start.files}
        assert max(errors.values())==0,(name,errors)
        assert read(study/name/'geometry.json')==fixed_geometry
        startup_differences[name]=errors
    report['exact_matched_startup_errors']=startup_differences
    report['limitations'].append('The frozen support gate requires at least 0.05 bodyweight of normal support load. Passing it is not proof that most weight has transferred to the surface or that the posture minimizes torque.')
    report['limitations'].append('Because the curriculum stopped at failed gates, earlier starts including standing were evaluated but never trained. Per-policy trained_start_stages are recorded separately.')
    grouped=[]
    for policy in [n for n in results if n.startswith('eval_')]:
        for baseline in ['fixed_all','reference_all']:
            for surface in [0,1]:
                for window in ['metrics_whole_trial','metrics_last10s','metrics_last5s']:
                    for metric in ['total_squared_torque_Nm2','total_absolute_torque_Nm','all_joint_rms_Nm','leg_squared_torque_Nm2','upper_squared_torque_Nm2']:
                        rows=[r for r in report['pairwise_effort'] if r['policy']==policy and r['baseline']==baseline and r['surface_id']==surface and r['window']==window and r['metric']==metric]
                        if rows:
                            pv=float(np.mean([r['policy_value'] for r in rows]));bv=float(np.mean([r['baseline_value'] for r in rows]))
                            grouped.append(dict(policy=policy,baseline=baseline,surface_id=surface,window=window,metric=metric,n=len(rows),stages=[r['start_stage'] for r in rows],policy_mean=pv,baseline_mean=bv,ratio_of_means_change_percent=100*(pv/bv-1),mean_paired_change_percent=float(np.mean([r['change_percent'] for r in rows]))))
    report['grouped_effort']=grouped
    report['passed']=True
    save(out/'validation.json',report);save(out/'results.json',results)
    rows=report['pairwise_effort']
    if rows:
        with (out/'paired_effort.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    render(out,report,results)
    print(json.dumps(dict(passed=True,transitions=all_total,evaluations={k:{p:v[p] for p in ['successes','physical_failures','standing_successes']} for k,v in report['evaluations'].items()}),indent=2))


def render(out,report,results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    names=list(results)
    labels={'fixed_all':'Fixed commands','reference_all':'Reference only'}
    for n in names:
        if n.startswith('eval_'):
            labels[n]=n.replace('eval_','Seed ').replace('_last_passing',' · last passed').replace('_last',' · last trained')
    matrix=np.array([[2 if r['success'] else 0 if any(r['failures'].values()) else 1 for r in results[n]['episodes']] for n in names])
    fig,ax=plt.subplots(figsize=(15,1.5+.62*len(names)))
    ax.imshow(matrix,cmap=ListedColormap(['#db927e','#ead4a1','#87c6a4']),vmin=0,vmax=2,aspect='auto')
    for i in range(len(names)):
        for j in range(14):ax.text(j,i,['physical\nfailure','no stable\nsupport','success'][matrix[i,j]],ha='center',va='center',fontsize=8)
    ax.set_yticks(range(len(names)),[labels[n] for n in names]);ax.set_xticks(range(14),[f'{"Seat" if j<7 else "Lean"}\n{j%7}' for j in range(14)])
    ax.axvline(6.5,color='white',lw=4);ax.set_title('Full 20-second outcome from every validated start\nStage 0 = standing; stage 6 = already supported; success requires the final 5 s of support')
    fig.tight_layout();fig.savefig(out/'stability.png',dpi=170);plt.close(fig)
    policies=[n for n in names if n.startswith('eval_')]
    fig,axes=plt.subplots(2,2,figsize=(13,7),squeeze=False)
    for b,baseline in enumerate(['fixed_all','reference_all']):
        for w,window in enumerate(['metrics_whole_trial','metrics_last5s']):
            ax=axes[w,b]
            for i,n in enumerate(policies):
                rows=[r for r in report['pairwise_effort'] if r['policy']==n and r['baseline']==baseline and r['window']==window and r['metric']=='total_squared_torque_Nm2']
                x=np.array([r['surface_id']*7+r['start_stage'] for r in rows])
                ax.plot(x+(i-(len(policies)-1)/2)*.09,[r['change_percent'] for r in rows],'o',label=labels[n],ms=6)
            ax.axhline(0,color='#777',lw=1);ax.set_xticks(range(14),[('S' if j<7 else 'L')+str(j%7) for j in range(14)],fontsize=8)
            ax.set_title(('Whole 20 s' if w==0 else 'Final 5 s')+' vs '+labels[baseline]);ax.set_ylabel('Total squared torque change (%)');ax.grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8);fig.suptitle('Effort on common full-duration support successes only\nNegative = less effort; missing cases failed one controller’s support test',fontsize=12)
    fig.tight_layout();fig.savefig(out/'paired_effort.png',dpi=170);plt.close(fig)
    table=[]
    for n in names:
        r=report['evaluations'][n]
        table.append(f'<tr><td>{html.escape(labels[n])}</td><td>{r["successes"]}/14</td><td>{r["physical_failures"]}/14</td><td>{r["standing_successes"]}/2</td></tr>')
    effort_table=[]
    for r in report['grouped_effort']:
        if r['window']=='metrics_last5s' and r['metric']=='total_squared_torque_Nm2':
            effort_table.append(f'<tr><td>{html.escape(labels[r["policy"]])}</td><td>{labels[r["baseline"]]}</td><td>{"Seat" if r["surface_id"]==0 else "Lean"}</td><td>{r["n"]}</td><td>{r["ratio_of_means_change_percent"]:+.2f}%</td></tr>')
    budgets=[]
    for seed,t in report['training'].items():
        state=t['state'];last=state['history'][-1]
        budgets.append(f'<li>Seed {seed}: {t["completed_updates"]} updates, {t["completed_transitions"]:,} transitions ({t["active"]:,} active + {t["warmup"]:,} reset warmup). Last trained stage {last["stage"]}; gate {last["successes"]}/2. Standing training reached: {state["reached_standing"]}.</li>')
    content=f'''<!doctype html><meta charset="utf-8"><title>Two-surface curriculum — verified results</title>
<style>body{{font:16px/1.5 system-ui;max-width:1120px;margin:40px auto;padding:0 24px;color:#173239}}h1{{font-size:30px}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #ccd6d8}}img{{width:100%;margin:20px 0}}.note{{background:#edf3f1;padding:18px}}</style>
<h1>Two-surface curriculum: verified comparison</h1>
<p>One level 40 cm seat and one 45° leaning surface. All 14 reset poses passed independent 20-second stationary validation. PPO learned bounded corrections around an offline reference motion, progressing from supported starts toward standing only after a 2/2 support gate.</p>
<p><a href="motion_replay.html">Inspect the saved motion with the interactive replay.</a></p>
<table><tr><th>Controller</th><th>Support successes</th><th>Physical failures</th><th>Standing successes</th></tr>{''.join(table)}</table>
<p>Fixed commands measure posture retention; an unsupported pose that remains standing is not a sitting or leaning success. The support gate requires at least 0.05 bodyweight of normal support load. Passing is not proof of complete weight transfer or minimum joint effort; the measured load and torques remain essential.</p>
<img src="stability.png"><ul>{''.join(budgets)}</ul>
<img src="paired_effort.png"><p>All 37 applied joint torques count. Squared Nm is an effort proxy; Nm RMS is the mean instantaneous all-joint RMS, not the RMS of averaged torques. Whole-trial, final 10-second and final 5-second comparisons, including leg and upper-body contributions, are in <a href="paired_effort.csv">paired_effort.csv</a>.</p>
<table><tr><th>Policy</th><th>Baseline</th><th>Surface</th><th>Common successes</th><th>Final 5 s squared torque change</th></tr>{''.join(effort_table)}</table><p>Changes in this table are ratios of means over the listed common successes. They are descriptive comparisons of repeated development conditions.</p>
<p class="note">Validation passed: {report['frozen_hashes_checked']} frozen source/bank hashes and {report['prior_hashes_checked']} preserved prior hashes; finite changed actor/critic weights; checkpoint and optimizer continuation; 32-update block accounting; exact evaluation starts; actual final checkpoint loads; full 20-second trials; constant fixed/finger commands and bounded learned commands.</p>
<ul>{''.join('<li>'+html.escape(x)+'</li>' for x in report['limitations'])}</ul>
<p>Detailed evidence: <a href="validation.json">validation.json</a> and <a href="results.json">results.json</a>. No policy is automatically promoted.</p>'''
    (out/'report.html').write_text(content)


if __name__=='__main__':
    main()
