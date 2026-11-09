"""Save CPU-only objective validation and the proposed experiment; launch no RL."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args=parser.parse_args()
out=Path(args.output).resolve()
out.mkdir(parents=True, exist_ok=False)
paths=[ROOT/'experiments/resting_posture_01.json',
       ROOT/'sitting_task/src/humanoid_sitting/resting_objective.py',
       ROOT/'scripts/test_resting_objective.py', Path(__file__).resolve()]
source=out/'source'; source.mkdir()
hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
for p in paths: shutil.copy2(p, source/p.name)
env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
test=subprocess.run([sys.executable, str(ROOT/'scripts/test_resting_objective.py'), '-v'],
                    cwd=ROOT, env=env, capture_output=True, text=True)
(out/'tests.log').write_text(test.stdout+test.stderr)
report=dict(utc=datetime.now(timezone.utc).isoformat(),
    status='cpu_objective_tests_passed' if test.returncode==0 else 'cpu_objective_tests_failed',
    test_returncode=test.returncode, hashes=hashes, training_started=False,
    simulator_validated=False, current_capped_study_modified=False,
    next_step='Implement and validate physically settled horizontal seated resets and the label/observation adapter')
(out/'status.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
sys.exit(test.returncode)
