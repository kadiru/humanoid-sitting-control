"""Saved-only results. Counts only the20 newly requested trials."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,csv,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/home/kadir/humanoid-sitting');RUN=ROOT/'outputs/furniture_examples_20261105_v2';SRC=ROOT/'experiments/furniture_examples_20261105_v2'
OUT=ROOT/'outputs/furniture_trials_review_20261105_v1'
sys.path.insert(0,str(SRC))
from audit_rollout import original
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True)
    status=read(ROOT/'outputs/furniture_examples_continuation_20261105_v1/results.json');assert status['complete']
    rows=[];bindings={};protocol=read(SRC/'protocol.json')
    for v in protocol['variants']:
        if v.get('reuse'):continue
        folder=RUN/v['id'];audit=read(folder/'independent_audit.json')
        bindings[str(folder/'independent_audit.json')]=sha(folder/'independent_audit.json')
        row=dict(id=v['id'],furniture=v['title'],task=v['task'],start=v['id'].split('_')[-1],
            passed=audit['admitted'],full20s=audit.get('full20s',False),final_hold_s=audit.get('final_hold_s'),longest_hold_s=None,
            first_failure_s=audit.get('first_failure_s'),slip_ticks=audit.get('slip_ticks'),illegal_ticks=audit.get('illegal_ticks'),
            source=str(folder),reason='',classification=audit['classification'])
        if not row['full20s']:
            row['reason']='Planner rejected start; motion trial did not run'
            log=RUN/(v['id']+'.log')
            if log.exists():
                bindings[str(log)]=sha(log)
                if 'self.nodes,self.t=self.plan.solve();assert self.t[-1]<14.4' in log.read_text():
                    row['reason']='Approach failed the14.4s timing guard; no motion rollout'
                    row['planning_guard']='teacher requires planned approach completion before14.4s'
                    row['planned_duration_s']=None
            row['details']=audit.get('details');rows.append(row);continue
        name='furniture_trajectory.npz' if v['task']=='sitting' else 'trajectory.npz'
        f=folder/name;assert sha(f)==audit['files'][name];bindings[str(f)]=audit['files'][name]
        with np.load(f) as z:d=dict(z)
        for extra in ['contact_mapping.json','furniture_capture.json','furniture_manifest.json','furniture_startup.json','results.json','manifest.json','startup.json']:
            p=folder/extra
            if p.exists():bindings[str(p)]=sha(p)
        r=original.physical(d);held=0;peak=0
        for good in r['good'][:,0]:held=held+1 if good and not audit['warm_failure'] else 0;peak=max(peak,held)
        assert abs(held*.02-row['final_hold_s'])<1e-10
        row['longest_hold_s']=peak*.02
        reasons=[]
        if audit['warm_failure']:reasons.append('standing reset failed')
        if not d['physical_joint_margin_ok'].all():reasons.append('joint safety margin crossed')
        if (d['wrong_force']>10).any():reasons.append('forbidden body contact')
        if row['illegal_ticks']:reasons.append('off-target or ambiguous contact')
        if row['slip_ticks']:reasons.append('loaded contact slipped')
        if (d['root_height']<.28).any() or (d['physical_trunk_tilt_rad']>np.deg2rad(55)).any():reasons.append('fall limit crossed')
        if (abs(d['physical_torque'])>d['physical_torque_scale']+.001).any():reasons.append('torque limit crossed')
        if (d['peak_bw']>2).any():reasons.append('impact limit crossed')
        if row['final_hold_s']<5:reasons.append('final hold shorter than5s')
        if not audit['mesh_mapping_validated']:reasons.append('contact geometry unvalidated')
        m={k[9:]:v for k,v in d.items() if k.startswith('physical_')}
        fn=(m['force_w']*m['normal_w']).sum(-1)
        ft=np.linalg.norm(m['force_w']-fn[...,None]*m['normal_w'],axis=-1)
        active=np.linalg.norm(m['force_w'],axis=-1)>.01*m['bodyweight_N'][...,None]
        util=np.where(ft<=1e-7,0,ft/np.maximum(m['friction']*np.maximum(fn,0),1e-7))
        load=(np.maximum(fn,0)*m['rest_patch']*m['permitted']).sum(-1)/m['bodyweight_N']
        blockers={'insufficient friction reserve':(active&(util>.90)).any(-1),
            'too little allowed support load':load<.05,'moving too fast':m['speed_m_s']>.1,
            'rotating too fast':m['angular_speed_rad_s']>.25,
            'tilt over hold limit':abs(m['trunk_tilt_rad'])>np.deg2rad(40)}
        row['hold_blockers_last_5s_ticks']={k:int(v[-250:,0].sum()) for k,v in blockers.items()}
        reasons.extend(k for k,v in row['hold_blockers_last_5s_ticks'].items() if v)
        row['reason']='Passed all original criteria' if row['passed'] else '; '.join(reasons)
        row['minimum_final_intended_load_bw']=audit['minimum_final_intended_load_bw']
        row['max_command_increment_rad']=audit['max_command_increment_rad']
        row['max_forbidden_body_contact_N']=float(d['wrong_force'].max())
        row['minimum_root_height_m']=float(d['root_height'].min())
        row['maximum_trunk_tilt_deg']=float(np.rad2deg(d['physical_trunk_tilt_rad']).max())
        row['peak_body_furniture_force_bw']=float((np.linalg.norm(d['physical_force_w'][:,:,:5],axis=-1).sum(-1)/d['physical_bodyweight_N']).max())
        startup=read(folder/('furniture_startup.json' if v['task']=='sitting' else 'startup.json'))
        row['initial_rear_camera_support_valid']=bool(startup['sensor_valid'][0])
        mapping=read(folder/'contact_mapping.json');mixed=0;seams=0;outside=0
        for frame in mapping['frames']:
            for labels in frame['mapped']:
                if not labels:continue
                normals=np.asarray([x['normal'] for x in labels])
                mixed+=int(np.any(np.linalg.norm(normals-normals[0],axis=1)>=1e-5))
                seams+=sum(x['reason']=='ambiguous_seam' for x in labels)
                outside+=sum(x['reason']=='outside_target_or_perimeter' for x in labels)
        row['contact_diagnostics']=dict(mixed_normal_channel_observations=mixed,ambiguous_seam_contact_points=seams,outside_target_contact_points=outside,
            counts_are_diagnostic_calls_not_control_ticks=True,physical_forbidden_body_contact=bool((d['wrong_force']>10).any()))
        if mixed and not audit['mesh_mapping_validated']:row['reason']+='; contact spans differently oriented faces'
        rows.append(row)
    assert len(rows)==20 and sum(r['task']=='sitting' for r in rows)==10
    summary=dict(new_trials=20,new_passes=sum(r['passed'] for r in rows),new_sitting_passes=sum(r['passed'] and r['task']=='sitting' for r in rows),new_leaning_passes=sum(r['passed'] and r['task']=='leaning' for r in rows),full20s_rollouts=sum(r['full20s'] for r in rows),reused_verified_leans_excluded=2,original72_retested=False,original_policies_modified=False,ppo_updates=0,criteria_unchanged=True,rows=rows,bindings=bindings,
        infrastructure_abort=dict(study='furniture_examples_20261105_v1',controls_before_reporter_error=1,warmup_controls=25,not_counted_as_completed_trial=True),gpu='4090 only;4070 untouched')
    summary['by_furniture']={name:dict(trials=len(selected),passes=sum(r['passed'] for r in selected),holds=[dict(id=r['id'],passed=r['passed'],final_hold_s=r['final_hold_s']) for r in selected]) for name in dict.fromkeys(r['furniture'] for r in rows) for selected in [[r for r in rows if r['furniture']==name]]}
    summary['planning_rejections_without_motion']=sum(not r['full20s'] for r in rows)
    summary['failed_complete_rollouts']=sum(r['full20s'] and not r['passed'] for r in rows)
    (OUT/'results.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    fields=['id','furniture','task','start','passed','full20s','final_hold_s','longest_hold_s','first_failure_s','slip_ticks','illegal_ticks','reason']
    with (OUT/'results.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    import html
    tr=''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in [i+1,r['furniture'],r['task'],r['start'],'PASS' if r['passed'] else ('FAIL' if r['full20s'] else 'NOT RUN'),'not run' if r['final_hold_s'] is None else f"{r['final_hold_s']:.2f}s",'not run' if r['longest_hold_s'] is None else f"{r['longest_hold_s']:.2f}s",r['reason']])+'</tr>' for i,r in enumerate(rows))
    (OUT/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>20 furniture trials</title><style>body{font:17px system-ui;background:#f5f7fa;color:#183444;margin:36px}video{width:100%;max-width:1200px;background:#111}table{border-collapse:collapse;width:100%;background:white}td,th{padding:12px;border-bottom:1px solid #dbe3e8;text-align:left}p{max-width:1100px}</style><h1>'+str(summary['new_passes'])+' passed / '+str(summary['failed_complete_rollouts'])+' failed / '+str(summary['planning_rejections_without_motion'])+' not run</h1><p>Reference-assisted demonstration controllers; no PPO training. All action limits and success criteria unchanged. A pass requires a full20s standing start, final5s continuous qualifying support, and no episode loaded slip or illegal contact. Hold times below are qualifying holds after all physical predicates and failure latching.</p><video controls src="furniture_trials_20.mp4"></video><p>Saved physical-pose replay at 2.5×; no simulation was rerun for this video. Starts rejected by the planner show static layouts only, with no hold time claimed. Mixed-normal contacts are unvalidated under the existing contact audit, even if the robot appears supported. Two older armchair passes are excluded. One earlier one-control reporting crash was preserved and corrected before this collection.</p><table><thead><tr><th>#</th><th>Furniture</th><th>Task</th><th>Start</th><th>Result</th><th>Final hold</th><th>Longest hold</th><th>Reason</th></tr></thead><tbody>'+tr+'</tbody></table><p><a href="results.csv">Download CSV</a> · <a href="results.json">Full audit summary</a></p>')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','bindings']}),flush=True)
if __name__=='__main__':main()
