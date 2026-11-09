"""Persistent, serial, predeclared PPO experiment and automatic evaluation."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import traceback

ROOT = Path('/home/kadir/humanoid-sitting')
STORE = Path('/data/humanoid-sitting-perception')
OUT = STORE / 'support_reward_study_01'
os.chdir(ROOT)
OUT.mkdir(exist_ok=False)
os.environ.update(OPENBLAS_NUM_THREADS='1', PYTHONUNBUFFERED='1')
completed = []


def status(phase, **kwargs):
    record = dict(phase=phase, utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  completed=completed, **kwargs)
    tmp = OUT/'status.tmp'; tmp.write_text(json.dumps(record, indent=2)+'\n')
    tmp.replace(OUT/'status.json')
    print(json.dumps(record), flush=True)


def run(name, arguments):
    command = ['bash', 'scripts/run.sh', 'python', '../scripts/train_support_reward_study.py',
        '--output', str(OUT/name), '--parent', str(STORE/'support_rl_independent_train_04/final.pt'),
        '--resume151', str(STORE/'support_split_ppo_train_01/final.pt'), '--interface', 'split',
        '--num-envs', '16', '--device', 'cuda:0',
        '--kit_args=--/renderer/multiGpu/activeCudaGpus=0, --/renderer/multiGpu/enabled=false', *arguments]
    (OUT/f'{name}.command.json').write_text(json.dumps(command, indent=2)+'\n')
    status('running', job=name, command=command)
    with (OUT/f'{name}.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    completed.append(name)


try:
    status('preflight')
    with (OUT/'unit_tests.log').open('w') as log:
        subprocess.run(['IsaacLab/.venv/bin/python', 'scripts/test_support_reward_study.py'],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    frozen = OUT/'frozen'; frozen.mkdir()
    old = json.loads((STORE/'support_split_ppo_frozen_01/manifest.json').read_text())
    for relative, digest in old.items():
        assert hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() == digest, relative
    paths = [ROOT/p for p in old]
    paths += [p for folder in ('scripts', 'sitting_task/src/humanoid_sitting')
              for p in (ROOT/folder).glob('*support_reward_study*') if p.is_file()]
    paths += [ROOT/'docs/SUPPORT_REWARD_STUDY_PROTOCOL.md']
    hashes = {}
    for path in sorted(set(paths)):
        relative = path.relative_to(ROOT); hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if path.suffix in ('.py', '.sh', '.md'):
            dest=frozen/relative; dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path,dest)
    (frozen/'manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    for arm in ('gentle', 'legacy'):
        run(f'train_{arm}', ['--mode','train','--reward-arm',arm,'--iterations','128',
            '--bank-from',str(STORE/'support_rl_pace_mixed_train_02/geometry_bank.json'),
            '--geometry-count','16','--geometry-seed','51621','--seed','52713'])
    banks = {
        'regression':['--bank-profile','legacy','--geometry-count','8','--geometry-seed','51403','--seed','51407'],
        'fresh':['--geometry-count','8','--geometry-seed','52641','--seed','52643'],
        'starts':['--bank-from',str(STORE/'support_rl_pace_mixed_train_02/geometry_bank.json'),
                  '--geometry-count','16','--geometry-seed','51621','--seed','52623'],
        'novel':['--geometry-count','8','--geometry-seed','52741','--seed','52743']}
    run('novel_initial', ['--mode','evaluate','--reward-arm','legacy',*banks['novel']])
    for arm in ('gentle', 'legacy'):
        for bank, arguments in banks.items():
            run(f'{bank}_{arm}', ['--mode','evaluate','--reward-arm','legacy',
                '--checkpoint',str(OUT/f'train_{arm}/final.pt'),*arguments])
    status('reporting')
    with (OUT/'report.log').open('w') as log:
        subprocess.run(['IsaacLab/.venv/bin/python','scripts/report_support_reward_study.py',
                        '--study',str(OUT)],stdout=log,stderr=subprocess.STDOUT,check=True)
    status('complete', report=str(OUT/'results/REPORT.md'))
except BaseException:
    status('failed', traceback=traceback.format_exc())
    raise
