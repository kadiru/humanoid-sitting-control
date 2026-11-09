"""Fail-closed approval and immutable-source checks. No scientific imports."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROPOSAL_SHA256 = '72c4aafa098419ba2ee69623ff5597b4fab14fd8626befb5a99a835f975253b2'
LIMIT = 40 * 1024**3
STAGES = {'A': (600., 420., 480., 598.), 'B': (600., 420., 480., 598.),
          'C': (45., 0., 30., 44.)}

def read(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4194304), b''):
            h.update(block)
    return h.hexdigest()

def validate_approval(receipt, freeze_sha):
    if receipt.get('execution_approved') is not True:
        raise PermissionError('This proposed cycle has not been approved.')
    assert receipt['proposal_sha256'] == PROPOSAL_SHA256
    assert receipt['freeze_sha256'] == freeze_sha
    assert receipt['explicit_human_message'] and receipt['human_message_reference']
    assert receipt['exceptions'] == ['single_cold_targeted_gap_retry',
        '600_600_45_seconds_including_audits', 'strengthened_problem_optimality']
    assert receipt['maximum_cycle_executions'] == 1

def authorize():
    # Called before creating the owner claim and again in every child.
    validate_approval(read(ROOT/'authorization.json'), sha(ROOT/'freeze.json'))
    f = read(ROOT/'freeze.json')
    assert f['ready'] and f['independent_source_review_complete']
    assert f['proposal_sha256'] == PROPOSAL_SHA256
    assert f['source_sha256']
    for name, expected in f['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    return f

def advance(stage, result, exit_record):
    """Pure decision rule used by the owner; no resumed or skipped stages."""
    if not exit_record['passed'] or not result['execution_and_audit_passed']:
        return None
    if not result['candidate_accepted']:
        return None
    return {'A': 'B', 'B': 'C', 'C': None}[stage]
