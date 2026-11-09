"""Inventory source/evidence without changing originals or printing secrets.

Run once before the initial import. This prepares explicit Git path groups and
SHA256 inventories; it never commits, pushes, deletes, or executes experiments.
"""
from __future__ import annotations

import collections
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'repository_metadata'
TEXT_EXT = {'.py', '.sh', '.json', '.yaml', '.yml', '.toml', '.md', '.txt',
            '.csv', '.js', '.css', '.usda', '.obj', '.xml', '.urdf', '.cpp',
            '.cu', '.cuh', '.h', '.hpp', '.cfg', '.ini', '.patch', '.diff',
            '.rst', '.ts', '.tsx', '.jsx', '.cmake', '.sql', '.ipynb'}
CODE_EXT = {'.py', '.sh', '.js', '.css', '.cpp', '.cu', '.cuh', '.h', '.hpp',
            '.patch', '.diff', '.ts', '.tsx', '.jsx', '.cmake'}
ASSET_EXT = {'.glb', '.gltf', '.bin', '.stl', '.ply', '.usd', '.usdc'}
SKIP_DIRS = {'.git', '__pycache__', '.venv', 'venv', 'node_modules',
             '.pytest_cache', '.mypy_cache', '.ruff_cache'}
SPECIAL = {'.gitignore', '.gitattributes', 'Makefile', 'Dockerfile', 'LICENSE',
           'NOTICE', 'requirements.txt'}
CURATED = {'rest_cycle_load_validation_20261109_v1/results/narrow_support_load_calibration.mp4',
           'rest_cycle_load_validation_20261109_v1/results/video_frame_249.jpg'}
SECRET_NAMES = {'.env', 'credentials', 'credentials.json', 'id_rsa', 'id_ed25519',
                '.netrc', 'hosts.yml'}
PATTERNS = {
    'private_key': re.compile(rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'github_token': re.compile(rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,})\b'),
    'aws_access_id': re.compile(rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'api_secret': re.compile(rb'\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{32,}\b'),
    'credential_url': re.compile(rb'https?://[^\s/:@]{2,80}:[^\s/@]{8,200}@'),
}
ASSIGNMENT = re.compile(rb'''(?i)(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[=:]\s*["']([^"'\s]{20,})["']''')

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def secret_findings(data):
    found = [name for name, pattern in PATTERNS.items() if pattern.search(data)]
    for match in ASSIGNMENT.finditer(data):
        value = match[1]
        frequencies = collections.Counter(value)
        entropy = -sum((n / len(value)) * math.log2(n / len(value)) for n in frequencies.values())
        if entropy > 3.7 and not any(x in value.lower() for x in [b'example', b'placeholder', b'your_', b'${', b'os.environ']):
            found.append('high_entropy_credential_assignment')
    return sorted(set(found))

def classify(path, size):
    rel = path.relative_to(ROOT).as_posix()
    if path.is_symlink():
        return False, 'symlink_target_not_followed'
    if path.name in SECRET_NAMES or (path.name.startswith('.env.') and path.name != '.env.example') or path.suffix in {'.pem', '.key', '.p12', '.pfx'}:
        return False, 'credential_filename'
    if path.name.startswith('._') or path.name == '.DS_Store' or path.suffix in {'.pyc', '.pyo'}:
        return False, 'cache_or_platform_metadata'
    if rel in CURATED:
        return True, 'curated_trial_media'
    allowed = path.suffix in TEXT_EXT | ASSET_EXT or path.name in SPECIAL
    if not allowed:
        return False, 'generated_or_unclassified_artifact'
    if size > 2 * 1024**2 and path.suffix not in CODE_EXT:
        return False, 'oversized_record_or_asset'
    return True, 'source_or_compact_record'

def group(rel):
    p = Path(rel)
    if p.parts[0] in {'repository_tools', 'repository_metadata'} or rel in {'.gitignore', '.gitattributes', 'README.md', 'ARTIFACTS.md'}:
        return 1
    if rel.startswith('ubuntu/') and not rel.startswith('ubuntu/experiments/'):
        return 2
    component = p.parts[2] if rel.startswith('ubuntu/experiments/') else p.parts[0]
    if component == 'rest_cycle_load_validation_20261109_v1' or rel == 'validated_low_seat_contact_01/current_successor_checkpoint.json':
        return 7
    if re.search(r'2026110[89]', component) or component.startswith('rest_cycle'):
        return 6
    if re.search(r'2026110[4-7]', component) or component.startswith(('furniture', 'box_armchair', 'armchair', 'camera_regions', 'capability_review')):
        return 5
    if re.search(r'2026102[6-9]', component) or component.startswith(('full_body', 'validated_contact_parallel', 'validated_contact_ppo', 'validated_contact_reward')):
        return 4
    return 3

def main():
    META.mkdir(exist_ok=True)
    groups = collections.defaultdict(list)
    counts = collections.Counter()
    sizes = collections.Counter()
    blocked = []
    exact_ignores = []
    rows = []
    for base, dirs, names in os.walk(ROOT, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(names):
            path = Path(base) / name
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith('repository_metadata/'):
                continue
            if path.is_symlink():
                target = os.readlink(path)
                size = len(os.fsencode(target))
                checksum = hashlib.sha256(os.fsencode(target)).hexdigest()
            else:
                size = path.stat().st_size
                checksum = digest(path)
            allowed, reason = classify(path, size)
            if allowed and path.suffix not in ASSET_EXT and rel not in CURATED:
                findings = secret_findings(path.read_bytes())
                if findings:
                    blocked.append({'path': rel, 'categories': findings})
                    allowed, reason = False, 'secret_screen_requires_review'
            action = 'track' if allowed else 'retain_external'
            counts[action] += 1
            sizes[action] += size
            rows.append((rel, size, checksum, action, reason))
            if allowed:
                groups[group(rel)].append(rel)
            elif reason in {'oversized_record_or_asset', 'generated_or_unclassified_artifact', 'symlink_target_not_followed', 'secret_screen_requires_review', 'credential_filename'}:
                exact_ignores.append('/' + rel.replace('[', '\\[').replace(']', '\\]').replace('*', '\\*').replace('?', '\\?'))
    marker = '# Exact oversized/generated/sensitive path exclusions are added by prepare_import.py.'
    ignore = ROOT / '.gitignore'
    prefix = ignore.read_text().split(marker)[0]
    ignore.write_text(prefix + marker + '\n' + '\n'.join(sorted(exact_ignores)) + '\n')
    # Hash the final ignore file, not its pre-generation contents.
    rows = [(p, ignore.stat().st_size, digest(ignore), action, reason)
            if p == '.gitignore' else (p, size, checksum, action, reason)
            for p, size, checksum, action, reason in rows]
    sizes = collections.Counter()
    for _, size, _, action, _ in rows:
        sizes[action] += size
    (META / 'artifact_inventory.tsv').write_text('path\tbytes\tsha256\tdisposition\treason\n' + ''.join('\t'.join(map(str, r)) + '\n' for r in rows))
    titles = {
        1: 'Initialize repository documentation and evidence preservation policy',
        2: 'Import Ubuntu task runtime, controllers, and baseline source',
        3: 'Preserve inherited low-seat research and validation records',
        4: 'Import PPO comparisons, imitation trainers, and reward validation',
        5: 'Import furniture perception, collision, load, and leaning work',
        6: 'Import complete-cycle planning, camera, and support-clearance work',
        7: 'Add spatial load sensing and record the failed narrow-support trial',
    }
    for i in range(1, 8):
        (META / f'commit_paths_{i}.txt').write_text(''.join(p + '\n' for p in groups[i]))
    summary = {'created_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'policy': 'all source; compact records <=2MiB; bulk artifacts preserved externally; no fabricated historical commit dates',
               'file_counts': dict(counts), 'bytes': dict(sizes), 'secret_screen_findings': blocked,
               'groups': [{'order': i, 'message': titles[i], 'files': len(groups[i])} for i in range(1, 8)]}
    (META / 'import_plan.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)
    if blocked:
        raise SystemExit('Review flagged files before staging; no secrets were printed.')

if __name__ == '__main__':
    main()
