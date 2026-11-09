"""Independently check a storage-only change before refreezing or any simulator launch."""
import ast
import hashlib
import json
import shutil
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OLD = HERE.parent/'full_body_contact_trainer_v1'
PRIOR = Path('/data/humanoid-sitting-perception/low_seat_contact_01/full_body_contact_imitation_v1')
OUT = Path('/home/kadir/humanoid-sitting/outputs/full_body_contact_imitation_v2')


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    before = read(OLD/'protocol.json'); after = read(HERE/'protocol.json')
    added = {'storage_revision', 'execution_deadline_unix', 'original_failed_attempt', 'storage_correction'}
    assert {k:v for k,v in after.items() if k not in added} == before
    assert after['execution_deadline_unix'] == read(PRIOR/'dependency_completed.json')['execution_deadline_unix']
    assert time.time() < after['execution_deadline_unix']
    prior = read(PRIOR/'status.json')
    assert prior['state'] == 'stopped' and not prior['active'] and not prior['completed'] and not prior['training_started']
    assert prior['failures'] and all(f['error'] == "AssertionError('DISK_RESERVE')" for f in prior['failures'])
    assert not list(PRIOR.glob('*.command.json')), 'A simulator child was previously launched'
    assert not (OUT/'receipt.json').exists(), 'New attempt already launched'
    assert read(OLD/'cpu_tests.json')['passed'] and read(OLD/'saved_reward_component_validation.json')['passed']
    checked = []
    for p in sorted(OLD.glob('*.py')):
        text = (HERE/p.name).read_text()
        if p.name in ['launch_after_comparison.py', 'prepare_launch.py']:
            text = text.replace("OUT=ROOT/'outputs/full_body_contact_imitation_v2'", "OUT=DATA/'full_body_contact_imitation_v1'")
        if p.name == 'launch_after_comparison.py':
            text = text.replace("deadline=protocol['execution_deadline_unix'];assert time.time()<deadline,'TOTAL_EXECUTION_WALL_CAP'", "deadline=time.time()+protocol['execution_wall_seconds']")
        if p.name == 'prepare_launch.py':
            text = text.replace("    inputs.update(DATA/'full_body_contact_imitation_v1'/name for name in ['status.json','receipt.json','dependency_completed.json'])\n", '')
        assert text == p.read_text(), ('Unexpected algorithm change', p.name)
        ast.parse((HERE/p.name).read_text())
        checked.append(p.name)
    for p in (OLD/'jobs').glob('*.json'):
        a=read(p); b=read(HERE/'jobs'/p.name); i=b['arguments'].index('--output')+1
        assert b['arguments'][i] == str(OUT/('capture_'+p.stem))
        b['arguments'][i]=a['arguments'][i]
        assert a==b, ('Teacher inputs changed',p.name)
    for path,h in read(OLD/'freeze.json').items():
        assert digest(Path(path))==h, ('Original frozen input changed',path)
    free=shutil.disk_usage(OUT.parent).free
    assert free > after['min_free_bytes']+after['max_output_bytes'], 'Insufficient space for full cap plus reserve'
    report={'passed':True,'scope':'Storage correction only; no simulator or optimizer execution',
            'all_original_frozen_files_match':True,'algorithm_sources_checked':checked,
            'teacher_commands_identical_except_output':True,'learning_and_physical_protocol_unchanged':True,
            'prior_simulator_children':0,'prior_ppo_updates':0,'original_deadline_preserved':after['execution_deadline_unix'],
            'output':str(OUT),'free_bytes':free,'required_full_cap_plus_reserve':after['min_free_bytes']+after['max_output_bytes'],
            'prior_status_sha256':digest(PRIOR/'status.json'),'automatic_retries':0,'physical_repair_claimed':False}
    (HERE/'storage_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':
    main()
