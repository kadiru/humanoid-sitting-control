"""Verify pre-existing source/checkpoints and record bounded new artifact storage."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--output', default='runs/shared_frame_integrity.json')
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
original = json.loads((root/'original_source_sha256.json').read_text())
changed = [name for name, digest in original.items()
           if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest]
baseline = root/'baselines/sensor-soft-landing-155-of-168'
expected = json.loads((baseline/'sha256.json').read_text())
baseline_changed = [name for name, digest in expected.items()
                    if hashlib.sha256((baseline/name).read_bytes()).hexdigest() != digest]
known = {
    'baselines/dagger-student-v1/best.pt': '4e187d9924c8b0357c9a0bde06a626df784af9c7005f67ae7dfaeecb5525e2e3',
    'runs/deployment_status_mask_05/student.pt': 'bdd8ee4a45955e7d27f37956047f2c2ecc68b709baebfb49e32780d276b8a2b1',
    'runs/contact_estimator_02/contact.pt': 'bd765e6390b186654d61896b34805a11e96b4d965b79e616a98f855205c84457',
}
checkpoint_changed = [name for name, digest in known.items()
                      if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest]
sizes = {str(folder.relative_to(root)): sum(p.stat().st_size for p in folder.rglob('*') if p.is_file() and not p.is_symlink())
         for folder in (root/'runs').glob('shared_frame*') if folder.is_dir()}
report = dict(original_source_files=len(original), original_source_changed=changed,
    archived_baseline_files=len(expected), archived_baseline_changed=baseline_changed,
    current_checkpoint_changed=checkpoint_changed, artifact_bytes=sizes,
    artifact_total_mib=sum(sizes.values())/2**20, storage_scope='regular files; sensor-only fixture symlink is not duplicated',
    free_gib=shutil.disk_usage(root).free/2**30,
    isaaclab_revision=subprocess.check_output(['git', '-C', str(root/'IsaacLab'), 'rev-parse', 'HEAD'], text=True).strip(),
    isaaclab_status=subprocess.check_output(['git', '-C', str(root/'IsaacLab'), 'status', '--short'], text=True))
(root/args.output).write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
assert not changed and not baseline_changed and not checkpoint_changed
assert report['isaaclab_status'] == ''
assert report['free_gib'] > 30.
