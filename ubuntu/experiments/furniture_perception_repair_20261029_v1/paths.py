import hashlib,json
from pathlib import Path
ROOT=Path('/home/kadir/humanoid-sitting')
HERE=ROOT/'experiments/furniture_perception_repair_20261029_v1'
OUT=ROOT/'outputs/furniture_perception_repair_20261029_v1'
ASSETS=ROOT/'experiments/furniture_test_20261028_v1'
PRIOR=ROOT/'experiments/motion_prior_ppo_20261029_v1'
PPO=ROOT/'outputs/motion_prior_ppo_20261029_v2'
CHECK=ROOT/'experiments/furniture_validation_20261028_v2'
BODY=ROOT/'experiments/motion_prior_ppo_20261029_v2'
GPU='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
PY=ROOT/'IsaacLab/.venv/bin/python'
def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def intent(case):
    return 'back' if case['approach']=='rear_back' else 'arm' if case['approach']=='arm' else 'seat' if case['approach'].startswith('seat') else 'top'
