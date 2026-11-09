"""Read-only scientific preparation: commands, dependency closure and hashes.

Writes only this new experiment's manifest files. Never starts a simulator.
"""
import ast
import platform
from pathlib import Path
import numpy as np
import torch
from common import ROOT, DATA, read, write, sha

HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/full_body_contact_imitation_v4'


def main():
    assert not (OUT/'receipt.json').exists()
    assert read(HERE/'cpu_tests.json')['passed']
    assert read(HERE/'saved_reward_component_validation.json')['passed']
    jobs=HERE/'jobs';jobs.mkdir(exist_ok=True);inputs=set()
    inputs.update(DATA/'full_body_contact_imitation_v1'/name for name in ['status.json','receipt.json','dependency_completed.json'])
    prior=ROOT/'outputs/full_body_contact_imitation_v2'
    inputs.update(prior/name for name in ['status.json','receipt.json','dependency_completed.json','build_dataset.log'])
    inputs.update(prior.glob('*.exit.json'))
    prior3=ROOT/'outputs/full_body_contact_imitation_v3'
    inputs.update(prior3/name for name in ['status.json','receipt.json','capture_known_old.log','capture_known_old.exit.json','capture_leaning.exit.json'])
    reuse=read(HERE/'leaning_reuse.json')
    inputs.update(Path(reuse['source'])/name for name in reuse['files'])
    def save_job(path,value):
        if path.exists():assert read(path)==value,'Existing frozen command changed'
        else:write(path,value)
    for name in ['known_old','known_recent','known_transfer']:
        source=DATA/'current_retention_09'/name/'candidate.command.json'
        original=read(source);assert original[2].endswith('run_low_seat_current_timing_probe.py')
        arguments=original[3:];assert arguments[arguments.index('--mode')+1]=='evaluate'
        arguments[arguments.index('--output')+1]=str(OUT/('capture_'+name))
        save_job(jobs/(name+'.json'),dict(entry='run_low_seat_current_timing_probe.py',arguments=arguments,task='sitting',bank=name,original_command_sha256=sha(source)))
        inputs.add(source)
    lean=read(HERE/'leaning_teacher_manifest.json')['args']
    keys=['bank','mode','surface','profiles','objective','start_stage','segment_seconds','num_envs','seed','device']
    arguments=[v for key in keys for v in ['--'+key.replace('_','-'),str(lean[key])]]
    arguments+=['--output',str(OUT/'capture_leaning'),'--kit_args='+lean['kit_args']]
    save_job(jobs/'leaning.json',dict(entry='run_transfer_effort.py',arguments=arguments,task='leaning',bank='leaning',original_manifest_sha256=sha(HERE/'leaning_teacher_manifest.json')))
    # Freeze exact argument-supplied input files, and files within named banks.
    for path in jobs.glob('*.json'):
        args=read(path)['arguments']
        for i,value in enumerate(args):
            if not value.startswith('/') or (i and args[i-1]=='--output'):continue
            p=Path(value);assert p.exists(),p
            if p.is_dir():inputs.update(x for x in p.rglob('*') if x.is_file())
            else:inputs.add(p)
    # Resolve local script imports statically, without executing old entrypoints.
    pending=[ROOT/'scripts'/name for name in ['run_low_seat_current_timing_probe.py','run_transfer_effort.py','run_supported_transition.py','run_supported_postures.py','run_resting_posture.py']]
    sources=set()
    while pending:
        path=pending.pop()
        if path in sources:continue
        sources.add(path);tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            names=[x.name for x in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                local=ROOT/'scripts'/(name.split('.')[0]+'.py')
                if local.is_file() and local not in sources:pending.append(local)
    sources.update(p for p in (ROOT/'sitting_task/src/humanoid_sitting').rglob('*.py') if not p.name.startswith('._'))
    sources.update([ROOT/'IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py',ROOT/'IsaacLab/source/isaaclab/isaaclab/utils/assets.py'])
    sources.update(ROOT/'IsaacLab/source/isaaclab/isaaclab/app'/name for name in ['sim_launcher.py','app_launcher.py'])
    sources.update(ROOT/'scripts'/x for x in ['run_supported_gpu0.sh','run_supported_gpu1_parallel_v1.sh'])
    inputs.add(ROOT/'runs/g1_kinematics_geometry_04/model.json')
    previous=read(DATA/'contact_parallel_gpu_v1/parallel_freeze_v2.json')
    assets=[Path(p) for p in previous if p.startswith(str(ROOT)+'/') and Path(p).suffix in ['.usd','.usda','.urdf']]
    inputs.update(assets)
    for path in [ROOT/'IsaacLab/uv.lock',ROOT/'IsaacLab/pyproject.toml',ROOT/'IsaacLab/.venv/pyvenv.cfg']:
        if path.exists():inputs.add(path)
    assert not any('/fresh' in str(p) or '/holdout' in str(p) for p in inputs),'NEW_HOLDOUT_NOT_AN_INPUT'
    write(HERE/'runtime.json',dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,cuda=torch.version.cuda,device='CPU preparation; no CUDA context created',scientific_steps=0))
    review=read(HERE/'source_review.json');assert review['passed'] and review['simulator_validation_complete'] is False
    write(HERE/'input_inventory.json',dict(script_dependencies=len(sources),inputs=len(inputs),assets=len(assets),new_holdout_opened=False,
          asset_note='G1_CFG retains its original remote USD URL. Asset configuration and frozen model are hashed; runtime names, masses and joint limits must match. Remote USD bytes are not claimed locally hashed.',
          imported_scripts=sorted(map(str,sources)),input_files=sorted(map(str,inputs))))
    own={p for p in HERE.rglob('*') if p.is_file() and p.suffix in ['.py','.json'] and p.name!='freeze.json'}
    freeze={str(p):sha(p) for p in sorted(sources|inputs|own)}
    write(HERE/'freeze.json',freeze)
    print(dict(frozen_files=len(freeze),script_dependencies=len(sources),inputs=len(inputs),simulator_steps=0))


if __name__=='__main__':main()
