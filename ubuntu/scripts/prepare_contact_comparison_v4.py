"""One diagnosed startup correction; inherit reward audit, preserve failed v3."""
from pathlib import Path
import json,hashlib,shutil,datetime,ast
ROOT=Path('/home/kadir/humanoid-sitting');D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
PRIOR=D/'contact_reward_comparison_128_v3';OUT=D/'contact_reward_comparison_128_v4'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,indent=2)+'\n')
assert read(PRIOR/'status.json')['state']=='stopped'
assert not (PRIOR/'identity_zero/weights_ready.json').exists()
assert not (PRIOR/'identity_zero/progress.json').exists()
assert len(read(PRIOR/'identity_zero/initial_settling.json'))==25
assert "'numpy.ndarray' object has no attribute 'cpu'" in (PRIOR/'identity_zero.log').read_text()
assert not OUT.exists();OUT.mkdir()
shutil.copytree(PRIOR/'training_bank',OUT/'training_bank')
for name in ['base_command.json','authorization.json','saved_reward_audit.json']:shutil.copy2(PRIOR/name,OUT/name)
p=read(PRIOR/'protocol.json');p['version']='contact_reward_comparison_128_v4'
p['recovery']={'prior':str(PRIOR),'failure':'NumPy env.index incorrectly used .cpu(); corrected to index.tolist(). No active trajectory or optimizer update completed.','classification':'Implementation correction, no reward/method/seed/action/physical criterion change; prior evidence preserved.','prior_initial_settling_transitions':600}
t=datetime.datetime.fromisoformat(read(PRIOR/'launch_receipt.json')['utc'].replace('Z','+00:00'))
p['charged_prior_seconds']=(datetime.datetime.now(datetime.timezone.utc)-t).total_seconds()
p['charged_prior_bytes']=sum(x.stat().st_size for x in PRIOR.rglob('*') if x.is_file())
p['budget']['initial_settling_transitions']+=600;p['budget']['total_physics_transitions']+=600
write(OUT/'protocol.json',p)
bindings=read(PRIOR/'freeze.json')
for q,h in bindings.items():assert sha(q)==h,q
for q in list((ROOT/'scripts').glob('*comparison*v4.py'))+[ROOT/'scripts/run_contact_learning_ppo_v4.py']:
 ast.parse(q.read_text());bindings[str(q)]=sha(q)
for q in list(OUT.glob('*.json'))+list((OUT/'training_bank').iterdir())+list(PRIOR.glob('*.json'))+[PRIOR/'identity_zero.log',PRIOR/'identity_zero/initial_settling.json']:
 bindings[str(q)]=sha(q)
write(OUT/'freeze.json',bindings)
write(OUT/'preparation.json',dict(passed=True,diagnosed_correction=True,no_previous_training_updates=True,prior_result=str(PRIOR/'status.json'),saved_reward_audit_inherited=True,original_wall_and_storage_caps_retained=True,prior_seconds=p['charged_prior_seconds'],prior_bytes=p['charged_prior_bytes']))
print(json.dumps(dict(prepared=str(OUT),bindings=len(bindings),prior_seconds=p['charged_prior_seconds'])),flush=True)
