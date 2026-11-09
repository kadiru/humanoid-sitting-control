"""Independent source/input review and freeze; never starts a simulator."""
import ast
import time
from pathlib import Path
from common import ROOT,DATA,read,write,sha

HERE=Path(__file__).resolve().parent
PRIOR=ROOT/'experiments/full_body_contact_trainer_4090_v4'
VALIDATED=ROOT/'outputs/full_body_contact_imitation_v4'


def node(p,name):return next(n for n in ast.walk(ast.parse(p.read_text())) if isinstance(n,ast.FunctionDef) and n.name==name)


def main():
    assert read(HERE/'step_tests.json')['passed']
    unchanged=['audit.py','common.py','environment.py','rewards.py','protected_sitting_cases.json']
    for f in unchanged:assert sha(HERE/f)==sha(PRIOR/f),f
    for name in ['command','advantages','__init__','normalize','choose','save','load']:
        assert ast.dump(node(HERE/'policy.py',name))==ast.dump(node(PRIOR/'policy.py',name)),name
    # Only diagnostic checkpoint/rollout save order changed outside the optimizer.
    before=(PRIOR/'run_training.py').read_text()
    expected=before.replace("                    assert metrics['actor_steps']>0\n",'').replace("                    from audit import weights", "                    assert metrics['actor_steps']>0\n                    from audit import weights",1)
    assert (HERE/'run_training.py').read_text()==expected
    p=read(HERE/'protocol.json');oldp=read(PRIOR/'protocol.json')
    for k in ['action','evaluation','curriculum','seeds','main_updates_per_seed_task','initialization',
              'actor_learning_rate','critic_learning_rate','ppo_clip','kl_reject','latent_std','discount','gae_lambda',
              'execution_deadline_unix','max_output_bytes','gpu_temperature_cap_C']:
        assert p[k]==oldp[k],k
    assert time.time()<p['execution_deadline_unix']
    s=(HERE/'policy.py').read_text()
    assert 'for exponent in range(8)' in s and 'alpha = 2.**(-exponent)' in s
    assert 'kl <= .02 and candidate_loss <= initial_loss' in s
    schedule=(HERE/'launch_after_comparison.py').read_text()
    assert schedule.count("assert gpu == 0, '4090_ONLY'")==2 and 'run_supported_gpu1_parallel_v1.sh' not in schedule
    assert 'ThreadPoolExecutor' not in schedule
    r=p['resource_accounting'];assert r['total_control_transitions_cap']==4688490+8208
    assert r['total_env_physics_transitions_cap']==4*r['total_control_transitions_cap']+32*72*1000
    assert r['simulator_processes']==32
    evidence={}
    def include(f):
        evidence[str(f)]=sha(f)
        return read(f) if f.suffix=='.json' else None
    assert include(VALIDATED/'dataset/dataset_audit.json')['passed']
    report={}
    for task in ['sitting','leaning']:
        for name in ['bank.npz','geometry.json','metadata.json']:include(VALIDATED/'dataset'/task/name)
        bank_hash=sha(VALIDATED/'dataset'/task/'bank.npz')
        reset=include(VALIDATED/f'preflight_{task}/reset_audit.json');assert reset['passed'] and reset['bank_sha256']==bank_hash
        reward=include(VALIDATED/f'teacher_replay_{task}/independent_audit.json');assert reward['passed'] and reward['strict_predicates_exact']
        for name,h in reward['files'].items():assert sha(VALIDATED/f'teacher_replay_{task}'/name)==h
        bc=[]
        for seed in [64111,64117]:
            a=include(VALIDATED/f'bc_{task}_{seed}/pretrain_audit.json');assert a['passed'] and a['bank_sha256']==bank_hash
            ck=VALIDATED/f'bc_{task}_{seed}/initial.pt';include(ck);assert sha(ck)==a['checkpoint_sha256'];bc.append(a)
        report[task]=dict(reset_passed=True,valid_by_stage=[sum(row[j] for row in reset['valid']) for j in range(4)],reward_audit_passed=True,
                          teacher_clean=reward['clean_count'],teacher_stable=sum(reward['stable']),cloning_seeds_passed=[x['seed'] for x in bc])
    for name in ['status.json','smoke_sitting.log','smoke_sitting.exit.json','smoke_sitting/initial.pt','smoke_sitting/status.json','smoke_sitting/updates.jsonl','smoke_sitting/results.json']:
        include(VALIDATED/name)
    failure=read(VALIDATED/'smoke_sitting/status.json');assert failure['actor_steps']==0 and failure['measured_kl']>.02
    initial=read(VALIDATED/'smoke_sitting/results.json');assert initial['simulator_control_steps']*72==8208
    write(HERE/'validation_reuse.json',dict(passed=True,tasks=report,files=evidence,original_failure=failure,
          replayed_simulation=False,reward_and_physical_code_identical=True))
    write(HERE/'source_review.json',dict(passed=True,unchanged_files={n:sha(HERE/n) for n in unchanged},
          optimizer_change=p['optimizer_step_revision'],resource_accounting=r,
          cpu_tests_sha256=sha(HERE/'step_tests.json'),all_previous_gates_reused_by_hash=True,
          main_weights_updating_verified=False,scientific_launches=0))
    bindings=read(PRIOR/'freeze.json')
    for f,h in bindings.items():assert sha(f)==h,f
    bindings.update(evidence)
    bindings.update({str(f):sha(f) for f in HERE.rglob('*') if f.is_file() and f.suffix in ['.py','.json'] and f.name!='freeze.json'})
    write(HERE/'freeze.json',bindings)
    print(dict(passed=True,frozen_files=len(bindings),validation=report,original_failed_proposal_kl=failure['measured_kl']))


if __name__=='__main__':main()
