"""Compare the audited PPO hold actor with the qualified touchdown controller."""

import argparse
from pathlib import Path
import sys

import numpy as np

from evaluate_touchdown_feedback import arguments, base_paths
from live_sitting_edge import worker
from repair_camera_fallback import KNOWN_GROUPS
from repair_live_touchdown import CHECKPOINT, HERE, PRIOR, ROOT, read, sha, write
from study_live_sitting import stage


HOLD_TICK = 750
BASELINE = ROOT/'outputs/touchdown_feedback_20261115_retained'
TRAINING = ROOT/'outputs/touchdown_feedback_ppo_20261115_v2'
BANKS = (*KNOWN_GROUPS, 'target_bank', 'adjustment_bank', 'untouched')
METRICS = ('hold_s', 'support_bodyweight', 'normalized_torque_squared', 'absolute_mechanical_power_w')


def evaluate_actor(checkpoint):
    sys.path.insert(0,str(HERE/'trainer'))
    import torch
    from policy import Learner

    trained = torch.load(checkpoint,map_location='cpu',weights_only=False)
    load_baseline = Learner.load
    choose_baseline = Learner.choose

    def load(self,path):
        baseline = load_baseline(self,path)
        for name in ('mean','scale'):
            torch.testing.assert_close(baseline[name].cpu(),trained[name],rtol=0,atol=0)
        assert int(trained['sitting']['hold_start_tick'])==int(self.sitting['hold_start_tick'])==HOLD_TICK
        self.ppo_actor_active = False
        return baseline

    def choose(self,obs,stochastic=True):
        assert not stochastic
        if not self.ppo_actor_active and round(float(obs[0,335])*1000)>=HOLD_TICK:
            assert torch.all((obs[:,335]*1000).round()==HOLD_TICK)
            self.actor.load_state_dict(trained['actor'])
            self.ppo_actor_active = True
            write(self.env.out/'ppo_actor.json',dict(first_tick=HOLD_TICK,checkpoint_sha256=sha(checkpoint)))
        return choose_baseline(self,obs,stochastic)

    Learner.load = load
    Learner.choose = choose
    worker()


def verify_audit(folder):
    audit_path = folder/'independent_audit.json'
    results = folder/'results.json'
    trace = folder/'trajectory.npz'
    audit = read(audit_path)
    assert audit['passed'] and audit['strict_predicates_exact'] and not audit['physical_tolerances_changed']
    assert audit['files']=={p.name:sha(p) for p in (trace,results)}
    return audit,read(results)['episodes']


def summaries(before,after):
    base,old = verify_audit(before)
    learned,new = verify_audit(after)
    assert [r['id'] for r in old]==[r['id'] for r in new]
    old_manifest,new_manifest = (read(path/'manifest.json') for path in (before,after))
    assert old_manifest['bank_sha256']==new_manifest['bank_sha256']
    assert old_manifest['args']['seed']==new_manifest['args']['seed']
    assert old_manifest['args']['bank']==new_manifest['args']['bank']
    assert new_manifest['action_limits_unchanged'] and new_manifest['physical_criteria_unchanged']
    with np.load(before/'trajectory.npz',allow_pickle=False) as a, np.load(after/'trajectory.npz',allow_pickle=False) as b:
        desired = [z['desired'] for z in (a,b)]
        commands = [z['command'] for z in (a,b)]
        prefix = np.abs(commands[1][:HOLD_TICK]-commands[0][:HOLD_TICK]).max(axis=(0,2))
        hold = np.abs(commands[1][HOLD_TICK:]-commands[0][HOLD_TICK:]).max(axis=(0,2))
        hold_desired = np.abs(desired[1][HOLD_TICK:]-desired[0][HOLD_TICK:]).max(axis=(0,2))
        support = [z['support_bw'][-250:].mean(0) for z in (a,b)]
    cases = []
    for i,(prior,current) in enumerate(zip(old,new,strict=True)):
        metrics = [
            dict(hold_s=row['final_hold_s'],support_bodyweight=float(load[i]),
                 normalized_torque_squared=audit['final_normalized_torque_squared'][i],
                 absolute_mechanical_power_w=audit['final_absolute_mechanical_power_W'][i])
            for row,load,audit in ((prior,support[0],base),(current,support[1],learned))
        ]
        cases.append(dict(
            id=prior['id'],touchdown_clean=prior['clean'],ppo_clean=current['clean'],
            touchdown=metrics[0],ppo=metrics[1],
            delta={k:metrics[1][k]-metrics[0][k] for k in METRICS},
            slip_ticks=[prior['slip_ticks'],current['slip_ticks']],
            illegal_ticks=[prior['illegal_ticks'],current['illegal_ticks']],
            physical_failures=current['physical_failures'],
            approach_max_command_delta_rad=float(prefix[i]),
            hold_max_command_delta_rad=float(hold[i]),
            hold_max_desired_delta_rad=float(hold_desired[i])))
    return cases


def aggregate(cases):
    common = [r for r in cases if r['touchdown_clean'] and r['ppo_clean']]
    means = {method:{key:float(np.mean([row[method][key] for row in common])) for key in METRICS}
             for method in ('touchdown','ppo')} if common else {}
    return dict(total=len(cases),touchdown_clean=sum(r['touchdown_clean'] for r in cases),
                ppo_clean=sum(r['ppo_clean'] for r in cases),
                improvements=[row['id'] for row in cases if row['ppo_clean'] and not row['touchdown_clean']],
                regressions=[row['id'] for row in cases if row['touchdown_clean'] and not row['ppo_clean']],
                common_passes=len(common),means_common_passes=means,
                approach_bitwise_equal_cases=sum(r['approach_max_command_delta_rad']==0 for r in cases),
                max_approach_command_delta_rad=max(r['approach_max_command_delta_rad'] for r in cases),
                cases_with_changed_hold_commands=sum(r['hold_max_command_delta_rad']>0 for r in cases),
                max_hold_command_delta_rad=max(r['hold_max_command_delta_rad'] for r in cases))


def compare(out):
    checkpoint = TRAINING/'training/final.pt'
    training_audit = TRAINING/'training/final_audit.json'
    initial = TRAINING/'training/initial.pt'
    audit = read(training_audit)
    assert audit['passed'] and audit['updates']==8 and audit['frozen_priors_unchanged']
    assert audit['checkpoint_sha256']==sha(checkpoint)
    assert read(TRAINING/'exit.json')['returncode']==0
    import torch
    base = torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
    warm = torch.load(initial,map_location='cpu',weights_only=False)
    for name in ('mean','scale'):
        torch.testing.assert_close(base[name],warm[name],atol=0,rtol=0)
    for name,parameter in base['actor'].items():
        torch.testing.assert_close(parameter,warm['actor'][name],atol=0,rtol=0)
    baseline_protocol = BASELINE/'paired_gate.json'
    design = read(baseline_protocol)
    assert design['passed']
    for name,digest in design['inputs'].items():
        assert sha(name)==digest,name
    gate_path = BASELINE/'ppo_gate.json'
    gate = read(gate_path)
    assert gate['passed'] and gate['known_clean']==212 and gate['repair_bank_clean']==[24,24]
    assert gate['fresh_clean']==24
    reference = {label:Path(gate['evidence'][label]) for label in BANKS}
    library,_ = base_paths()
    config = BASELINE/'feedback.json'
    guard = BASELINE/'protected.npz'
    extra,inputs = arguments(library,config,guard,Path(design['routes']))
    inputs.extend((checkpoint,training_audit,initial,HERE/'live_sitting_edge.py',baseline_protocol,gate_path,
                   BASELINE/'retention.json'))
    assert all(sha(p)==design['inputs'][str(p)] for p in (config,guard))
    sources = [sha(HERE/'live_sitting_edge.py'),sha(HERE/'touchdown_feedback.py')]
    assert all(digest in design['inputs'].values() for digest in sources)
    out.mkdir()
    reservation = out/'reservation.json'
    reservation.write_bytes((BASELINE/'reservation.json').read_bytes())
    protocol = out/'comparison_protocol.json'
    write(protocol,dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
          actor_switch_tick=HOLD_TICK,baseline={k:str(v) for k,v in reference.items()},
          original_case_banks=True,matched_saved_baseline=True,scoring_unchanged=True,action_limits_unchanged=True))
    all_cases = {}
    results = out/'comparison.json'
    for label,reference_path in reference.items():
        audit,rows = verify_audit(reference_path)
        assert audit['clean_count']==len(rows)
        stage(out,label,Path(design['banks'][label]),PRIOR,worker=Path(__file__),
              extra=['worker',str(checkpoint),*extra],protocol=protocol,inputs=inputs)
        assert read(out/label/'ppo_actor.json')==dict(first_tick=HOLD_TICK,checkpoint_sha256=sha(checkpoint))
        all_cases[label] = summaries(reference_path,out/label)
        write(results,dict(banks={k:aggregate(v) for k,v in all_cases.items()},cases=all_cases))
    known = [r for label in KNOWN_GROUPS for r in all_cases[label]]
    assert len(known)==212 and all(len(all_cases[k])==24 for k in BANKS[-3:])
    data = read(results)
    data.update(known=aggregate(known),overall=aggregate([r for v in all_cases.values() for r in v]))
    assert data['overall']['total']==284 and data['overall']['cases_with_changed_hold_commands']>0
    write(results,data)
    write(out/'status.json',dict(state='complete',ppo_clean=data['overall']['ppo_clean'],total=284))


if __name__=='__main__':
    if sys.argv[1:2]==['worker']:
        checkpoint_arg = Path(sys.argv[2])
        del sys.argv[1:3]
        evaluate_actor(checkpoint_arg)
    else:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument('output',type=Path)
        compare(parser.parse_args().output.resolve())
