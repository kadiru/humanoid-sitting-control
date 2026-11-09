"""Saved-output accounting and preservation only; does not repeat scientific arithmetic."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

source = Path('/home/kadir/humanoid-sitting/scripts/consistency_ray_diagnostic_20261022_v1')
out = Path('/data/humanoid-sitting-perception/low_seat_contact_01/consistency_ray_diagnostic_20261022_v1')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_text())
def write(name, value):
    with (out / name).open('x') as f:
        json.dump(value, f, indent=2)
        f.write('\n')

sys.path.insert(0, str(source))
import common
workers = common.workers({os.getpid(), os.getppid()})
assert not workers
assert sha(source / 'contract.json') == '8933f47dc3bd8dc78ccdf82bc6878c49ef767a7ac06e2ce3935f6bf8fab58c1e'
assert sha(source / 'authorization.json') == '80975f168351bc81af7e4eae86b512dde1f599f87f7c3908cef40fc525c73606'
approval = read(source / 'approval_receipt_20261022_v1.json')
for name, digest in approval['source_sha256'].items():
    assert sha(source / name) == digest, name
candidate = read(out / 'candidate.json')
audit = read(out / 'audit.json')
result = read(out / 'result.json')
execution = read(source / 'execution_receipt.json')
assert candidate['diagnostic_passed'] and audit['passed'] and audit['diagnostic_passed']
assert result['diagnostic_passed'] and result['execution_and_audit_passed']
assert execution['actual_supervisor_exit'] == 0 and execution['outer_within_cap']
assert execution['launches'] == 1
assert all(s['actual_exit'] == 0 and s['completed_within_deadline'] for s in result['stages'])
assert sha(out / 'raw_direction.npy') == audit['raw_sha256']
assert sha(out / 'constructed_point.npy') == audit['point_sha256']
utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
summary = {
    'utc': utc,
    'classification': 'Completed one-shot constructive primal/model-descent diagnostic; exact original feasibility passed; no controller admission',
    'approved_contract_sha256': sha(source / 'contract.json'),
    'authorization_sha256': sha(source / 'authorization.json'),
    'human_approval_question': approval['human_approval_question'],
    'human_answer': approval['human_answer'],
    'approval_consumed': True, 'no_retry': True, 'launches': 1,
    'diagnostic_passed': True, 'exact_stored_binary64_primal_feasible': True,
    'original_inequality_rows': audit['full_original_inequality_rows'],
    'original_equality_rows': audit['original_equality_rows'],
    'original_bounds_checked': audit['original_bounds_checked'],
    'raw_consistency_rows': audit['raw_consistency_rows'],
    'exact_products': audit['exact_products'],
    'linear_block_systems_completed': 1365,
    'linear_block_shape': [15, 15],
    'LP_solves': 0, 'nonlinear_root_calls': 0,
    'alpha': candidate['alpha'], 'dyadic_exponent': candidate['dyadic_exponent'],
    'model_merit_before': candidate['mu'],
    'floating_modeled_merit_after': candidate['modeled_theta'],
    'floating_modeled_reduction': candidate['modeled_reduction'],
    'epigraph': candidate['epigraph'],
    'exact_epigraph_reduction_numerator_and_denominator_exponent': audit['exact_epigraph_reduction'],
    'correction_error_outward_upper_bound': candidate['correction_error'],
    'point_sha256': audit['point_sha256'], 'raw_sha256': audit['raw_sha256'],
    'independent_scientific_audit_passed': True,
    'worker_actual_exits': {s['stage']: s['actual_exit'] for s in result['stages']},
    'supervisor_actual_exit': execution['actual_supervisor_exit'],
    'outer_launch_through_exit_elapsed_s': execution['outer_launch_through_exit_elapsed_s'],
    'supervisor_reported_elapsed_s': result['total_elapsed_s'],
    'total_cap_s': 45, 'outer_overshoot_s': execution['outer_overshoot_s'],
    'peak_worker_rss_bytes': max(s['peak_rss_bytes'] for s in result['stages']),
    'address_space_limit_bytes_per_worker': result['address_space_bytes'],
    'primary_optimum_claimed': False, 'full_KKT_passed': False,
    'interior_clearance_checked': False, 'candidate_accepted': False,
    'finite_maps': 0, 'new_model_calls': 0, 'new_J_calls': 0,
    'physical_trials': 0, 'PPO_updates': 0, 'holdout_trials': 0,
    'physical_admission': False, 'physical_success': False,
    'meaning': 'A fresh coordinated correction admits a very small dyadic step satisfying every stored original affine constraint exactly and reducing the epigraph. The floating modeled reduction is only about 3.66e-9. This does not establish a useful finite encoded update, primary optimality, interior clearance, nonlinear improvement, executable motion or repaired controller.',
    'last_accepted_research_iterate': 'finite_model_step_20261020_v1 remains unchanged; its actual numerical motion gate failed',
    'physical': {'clean': '64/72', 'stable': 70, 'slipping': 8, 'housing_contact': 2,
                 'categories_overlap': True, 'all64_protected': True, 'all8_unresolved': True,
                 'main_PPO': 'unstarted', 'fresh_holdout': 'closed'},
    'workers_at_seal': workers, 'task_complete': False, 'automation': 'ACTIVE',
    'next_work': 'Assess whether this tiny modeled descent can support a substantive next construction under all remaining primary/interior/finite/model/physical gates. No repeat, step enlargement, encoding, simulation or PPO is authorized by this consumed diagnostic approval.'
}
write('summary.json', summary)
write('accounting_review.json', {
    'utc': utc, 'scope': 'Saved-result, exit, timing, source and output-hash review only; scientific checker not rerun',
    'passed': True, 'scientific_audit_scope': audit['scope'], 'workers': workers,
    'all_actual_exits_zero': True, 'outer_elapsed_includes_process_startup_and_shutdown': True,
    'within_45_seconds': True, 'scientific_source_unchanged_since_approval': True,
    'read_only_orientation_failure': 'Local combined read found no manifest.json in the preceding minimax folder; rg confirmed the existing filename is SHA256.json. No scientific work was affected.'
})
(out / 'result_explained.txt').write_text(
    'The approved diagnostic passed in 3.444 seconds including startup, independent audit and shutdown.\n'
    'All 652529 original inequalities, 1366 equalities and original bounds passed the independent exact dyadic check.\n'
    'The fixed correction allowed alpha=2^-25; floating modeled mismatch decreased by 3.6568856276453232e-9.\n'
    'The exact epigraph reduction is 131753189 / 2^56. This is a very small modeled step.\n'
    'No primary optimum, interior clearance, finite encoding, nonlinear validation, physical motion or PPO was established.\n'
    'The one-time approval is consumed. No retry or automatic downstream execution.\n'
    'Best09 stays 64/72 clean, 70 stable, 8 slipping, 2 housing contact; categories overlap. All eight failures remain unresolved.\n')
shutil.copytree(source, out / 'source_snapshot', ignore=shutil.ignore_patterns('__pycache__'))
entries = {}
for f in sorted(out.rglob('*')):
    if f.is_file():
        entries[str(f.relative_to(out))] = {'sha256': sha(f), 'bytes': f.stat().st_size}
write('SHA256.json', {'scope': 'All sealed files except this manifest; external inputs/runtime remain bound by source_snapshot/contract.json', 'files': entries})
for root in (out, source):
    for f in root.rglob('*'):
        if f.is_file(): f.chmod(0o444)
    for f in sorted(root.rglob('*'), reverse=True):
        if f.is_dir(): f.chmod(0o555)
    root.chmod(0o555)
print(json.dumps({'summary_sha256': sha(out / 'summary.json'), 'manifest_sha256': sha(out / 'SHA256.json'),
                  'manifest_entries': len(entries), 'bytes_manifested': sum(v['bytes'] for v in entries.values()),
                  'workers': workers, 'utc': utc, 'scientific_arithmetic_repeated': False}))
