"""Read-only source/input/resource audit; no simulator launch or optimization."""
import ast
import hashlib
import json
import shutil
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'full_body_contact_trainer_storage_v2'
PRIOR=Path('/home/kadir/humanoid-sitting/outputs/full_body_contact_imitation_v2')
OUT=Path('/home/kadir/humanoid-sitting/outputs/full_body_contact_imitation_v3')
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    prior=read(PRIOR/'status.json')
    assert prior['state']=='stopped' and not prior['active'] and not prior['training_started']
    names=['known_old','known_recent','known_transfer','leaning']
    assert set(prior['completed'])=={'capture_'+n for n in names}
    assert read(PRIOR/'build_dataset.exit.json')['returncode']==1
    for n in names:
        assert read(PRIOR/('capture_'+n+'.exit.json'))['returncode']==0
        assert not (PRIOR/('capture_'+n)/'demonstrations_50hz.npz').exists()
    p=read(HERE/'protocol.json'); old=read(OLD/'protocol.json')
    excluded={'capture_revision','recovery_accounting','resource_accounting','gpu_availability_wait','stop_behavior'}
    assert {k:v for k,v in p.items() if k not in excluded}=={k:v for k,v in old.items() if k not in excluded}
    assert p['stop_behavior']==old['stop_behavior'].replace('occupied GPU at handoff or ','')+' Occupied GPUs wait without interruption only until the original execution deadline.'
    assert p['execution_deadline_unix']==read(PRIOR/'dependency_completed.json')['execution_deadline_unix']
    assert time.time()<p['execution_deadline_unix']
    r=p['resource_accounting'];o=old['resource_accounting']
    assert r['total_control_transitions_cap']==o['total_control_transitions_cap']+92250
    assert r['simulator_processes']==o['simulator_processes']+4==30
    assert r['total_env_physics_transitions_cap']==r['total_control_transitions_cap']*4+30*72*1000
    retained=['audit.py','build_dataset.py','common.py','environment.py','policy.py','pretrain.py','rewards.py','run_training.py','validate_saved.py']
    for name in retained:assert (HERE/name).read_bytes()==(OLD/name).read_bytes(),name
    def sampler(path):
        tree=ast.parse(path.read_text())
        return ast.dump(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='step'),include_attributes=False)
    assert sampler(HERE/'capture_teacher.py')==sampler(OLD/'capture_teacher.py')
    for n in names:
        a=read(OLD/'jobs'/(n+'.json'));b=read(HERE/'jobs'/(n+'.json'));i=b['arguments'].index('--output')+1
        assert b['arguments'][i]==str(OUT/('capture_'+n));b['arguments'][i]=a['arguments'][i];assert a==b
    for path,h in read(OLD/'freeze.json').items():assert sha(Path(path))==h,path
    assert read(HERE/'capture_shutdown_tests.json')['passed']
    assert not OUT.exists()
    assert shutil.disk_usage(OUT.parent).free>p['min_free_bytes']+p['max_output_bytes']
    report={'passed':True,'unchanged_learning_and_physical_sources':retained,'unchanged_per_step_capture_sampler':True,
            'teacher_commands_unchanged_except_output':True,'runtime_shutdown_test_passed':True,
            'prior_failed_captures_charged':4,'total_process_cap_including_failed_captures':30,
            'total_control_transition_cap':r['total_control_transitions_cap'],'original_wall_deadline':p['execution_deadline_unix'],
            'original_deadline_extended':False,'memory_storage_and_temperature_caps_unchanged':True,
            'new_simulator_steps':0,'new_ppo_updates':0,'runtime_demonstration_validation_passed':False}
    (HERE/'capture_correction_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
