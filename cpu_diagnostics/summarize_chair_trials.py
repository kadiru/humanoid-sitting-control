"""Join scored chair trials with the measured touchdown and camera events."""

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path

import numpy as np

from compare_touchdown_ppo import verify_audit
from run_repaired_sitting_ppo import read, write


def first(mask):
    hits = np.flatnonzero(mask)
    return float(hits[0]+1)*.02 if len(hits) else None


def collect(studies):
    all_rows = []
    receipts = []
    for study in studies:
        recorded = {r['id']:r for r in csv.DictReader((study/'results.csv').open(newline=''))}
        remaining = set(recorded)
        for path in sorted(study.rglob('independent_audit.json')):
            episode = path.parent
            audit,rows = verify_audit(episode)
            manifest = read(episode/'manifest.json')
            assert manifest['action_limits_unchanged'] and manifest['physical_criteria_unchanged']
            scene = episode/'scene_assets_verified.json'
            if scene.exists():
                assert [r['id'] for r in read(scene)]==[r['id'] for r in rows]
            with np.load(episode/'trajectory.npz',allow_pickle=False) as trace, \
                 np.load(episode/'edge_audit.npz',allow_pickle=False) as edges, \
                 np.load(episode/'camera_touchdown_selection.npz',allow_pickle=False) as camera:
                recover = {k:trace[k] for k in ('support_bw','no_slip','legal','failure','wrong_force',
                                                'command','previous')}
                assert np.abs(recover['command']-recover['previous']).max()<.0030002
                early = edges['values'][edges['ticks']<=310]
                edge_m = early[:,:,0]
                edge_error = np.abs(edge_m-early[:,:,1])
                selected = camera['selected']
                tags = camera['ids']
                for i,r in enumerate(rows):
                    key = r['id']
                    assert key in remaining
                    remaining.remove(key)
                    row = recorded[key]
                    assert (row['status']=='pass')==r['clean']
                    finite = np.isfinite(edge_m[:,i])
                    all_rows.append(dict(row,cohort=study.name,trajectory=str(episode/'trajectory.npz'),
                        active_controls=r['steps'],first_support_s=first(recover['support_bw'][:,i]>.05),
                        first_slip_s=first(~recover['no_slip'][:,i]),first_illegal_s=first(~recover['legal'][:,i]),
                        first_hard_failure_s=first(recover['failure'][:,i]),
                        first_wrong_force_over_10n_s=first(recover['wrong_force'][:,i]>10.),
                        peak_wrong_force_n=float(recover['wrong_force'][:,i].max()),
                        final_support_bw=float(np.mean(recover['support_bw'][-250:,i])),
                        early_edge_frames=int(finite.sum()),
                        median_edge_error_mm=float(1000*np.median(edge_error[finite,i])) if finite.any() else None,
                        selected_camera_route=str(tags[selected[i]]) if selected[i]>=0 else ''))
            receipts.append(dict(output=str(episode),trajectory_sha256=audit['files']['trajectory.npz'],
                                 cases=len(rows),colliders_verified=scene.exists()))
        for key in sorted(remaining):
            assert recorded[key]['status'] in ('startup failure','depth unavailable'),key
            all_rows.append(dict(recorded[key],cohort=study.name))
    return all_rows,receipts


def summarize(rows):
    total = Counter(r['status'] for r in rows)
    groups = defaultdict(list)
    primary_pairs = defaultdict(dict)
    for r in rows:
        groups[r['family']].append(r)
        if r['id'].startswith('batch_'):
            batch,placement,family,shift = r['id'].split('/')
            primary_pairs[batch,placement,shift][family] = r['status']
    families = {}
    for name,subset in groups.items():
        states = Counter(r['status'] for r in subset)
        passing = [float(r['power_w']) for r in subset if r['status']=='pass']
        paired = [(g[name]=='pass',g['plain_plate']=='pass') for g in primary_pairs.values()
                  if name!='plain_plate' and g.get(name) in ('pass','fail')
                  and g.get('plain_plate') in ('pass','fail')]
        families[name] = dict(states=states,slips=sum(int(r['slip_ticks'] or 0)>0 for r in subset),
            forbidden=sum(r['forbidden_contact']=='True' for r in subset),
            measured_edges=sum(r.get('early_edge_frames',0)>0 for r in subset),
            selected_routes=sum(bool(r.get('selected_camera_route')) for r in subset),
            mean_passing_power_w=float(np.mean(passing)) if passing else None,
            paired_plate=dict(total=len(paired),chair_only=sum(a and not b for a,b in paired),
                              plate_only=sum(b and not a for a,b in paired)))
    return dict(total=len(rows),states=total,families=families)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('studies',type=Path,nargs='+')
    args = parser.parse_args()
    cases,receipts = collect(args.studies)
    assert cases and len(cases)==len({r['id'] for r in cases})
    args.output.mkdir()
    with (args.output/'chair_trials.csv').open('x',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in cases for k in r)))
        writer.writeheader()
        writer.writerows(cases)
    times = [read(p)['wall_s'] for s in args.studies for p in s.rglob('evaluation.exit.json')]
    write(args.output/'summary.json',dict(**summarize(cases),gpu_worker_wall_s=sum(times),
                                         audited_traces=receipts))
