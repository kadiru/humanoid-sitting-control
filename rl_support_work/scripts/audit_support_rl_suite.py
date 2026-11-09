"""Verify saved source provenance and completeness of the evaluated rollouts."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(); p.add_argument('--suite',required=True); a=p.parse_args()
root=Path(__file__).resolve().parents[1]; suite=Path(a.suite)
report=json.loads((suite/'results.json').read_text())
store=suite.parent; checks=[]
names=list(report['runs'].values())+[Path(t['path']).name for t in report['training'].values()]
for name in names:
    run=store/name; manifest=json.loads((run/'manifest.json').read_text())
    hashes=manifest['hashes']; archived=0
    for relative,expected in hashes.items():
        assert hashlib.sha256((root/relative).read_bytes()).hexdigest()==expected,(name,relative,'current source mismatch')
        if relative.endswith('.py'):
            assert hashlib.sha256((run/'source'/Path(relative).name).read_bytes()).hexdigest()==expected,(name,relative,'archive mismatch')
            archived+=1
    if 'ablation' in manifest:
        filename='evaluate_support_rl_without_live_depth.py'
        expected=manifest['ablation']['ablation_source_sha256']
        assert hashlib.sha256((run/'source'/filename).read_bytes()).hexdigest()==expected
        assert hashlib.sha256((root/'scripts'/filename).read_bytes()).hexdigest()==expected
        archived+=1
    result=json.loads((run/'results.json').read_text())
    trace_samples=None
    if result['mode']!='train':
        assert len(result['episodes'])==manifest['arguments']['num_envs']
        with np.load(run/'rollout.npz') as z:
            assert len(z['t'])==200 and abs(z['t'][-1]-20)<1.e-5
            assert np.all(np.diff(z['t'])>0)
            assert z['live'].shape==(200,manifest['arguments']['num_envs'],10)
            assert np.isfinite(z['live']).all() and np.isfinite(z['body']).all()
            trace_samples=int(z['live'].shape[0]*z['live'].shape[1])
    checks.append(dict(run=name,current_hashes_verified=len(hashes),archived_sources_verified=archived,
                       complete_episodes=len(result['episodes']),recorded_case_frames=trace_samples))
for data in report['training'].values():
    assert hashlib.sha256((Path(data['path'])/'final.pt').read_bytes()).hexdigest()==data['final_checkpoint_sha256']
    assert data['verification']['finite'] and data['verification']['max_parameter_change']>0
    assert data['verification']['optimizer_contains_actor']
integrity=json.loads((store/'support_rl_suite_integrity_02.json').read_text())
assert not integrity['original_source_changed'] and not integrity['archived_baseline_changed']
assert not integrity['current_checkpoint_changed'] and integrity['isaaclab_status']==''
video=suite/'sitting_comparison.mp4'
result=dict(passed=True,runs=checks,original_source_files=integrity['original_source_files'],
    archived_baseline_files=integrity['archived_baseline_files'],
    video_sha256=hashlib.sha256(video.read_bytes()).hexdigest(),
    evaluated_trace_case_frames=sum(c['recorded_case_frames'] or 0 for c in checks),
    regular_file_bytes=sum(p.stat().st_size for n in names for p in (store/n).rglob('*') if p.is_file() and not p.is_symlink()))
with (suite/'bundle_audit.json').open('x') as f:json.dump(result,f,indent=2)
print(json.dumps(result,indent=2))
