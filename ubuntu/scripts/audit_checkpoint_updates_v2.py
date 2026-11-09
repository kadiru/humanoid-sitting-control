"""Verify actual finite parameter changes and Adam steps in a saved PPO checkpoint."""
import hashlib
import json
import sys
from pathlib import Path
import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(folder, name):
    folder = Path(folder)
    initial = torch.load(folder/'initial.pt', map_location='cpu', weights_only=False)
    final = torch.load(folder/name, map_location='cpu', weights_only=False)
    delta = final['updates'] - initial['updates']
    assert delta > 0
    records = json.loads((folder/'learning_audit.json').read_text())[:delta]
    assert len(records) == delta
    assert records[-1]['updates'] == final['updates']
    report = {}
    for model in ['actor', 'critic']:
        before, after = initial[model+'_state_dict'], final[model+'_state_dict']
        assert set(before) == set(after)
        assert all(torch.isfinite(v).all() for v in after.values())
        changes = {k: float((after[k]-v).abs().max()) for k, v in before.items()}
        assert max(changes.values()) > 0, ('NO_PARAMETER_CHANGE', model)
        optimizer = final[model+'_optimizer']['state']
        old_optimizer = initial[model+'_optimizer']['state']
        expected_steps = sum(r['optimizer'][model+'_steps'] for r in records)
        assert optimizer and expected_steps > 0
        actual_steps = []
        for key, state in optimizer.items():
            assert all(torch.isfinite(v).all() for v in state.values() if torch.is_tensor(v))
            old_step = old_optimizer.get(key, {}).get('step', 0)
            actual_steps.append(int(state['step']) - int(old_step))
        assert set(actual_steps) == {expected_steps}, (model, actual_steps, expected_steps)
        assert all(all(isinstance(v, (int, float)) and torch.isfinite(torch.tensor(v))
                       for v in r['optimizer'].values()) for r in records)
        report[model] = dict(max_abs_parameter_change=max(changes.values()),
                             changed_tensors=sum(v > 0 for v in changes.values()),
                             adam_steps_per_parameter=expected_steps, finite=True)
    result = dict(passed=True, checkpoint=name, initial_sha256=sha(folder/'initial.pt'),
                  checkpoint_sha256=sha(folder/name), local_updates=delta,
                  models=report, physical_success_claimed=False)
    target = folder/(Path(name).stem+'_weight_audit.json')
    if target.exists():
        assert json.loads(target.read_text()) == result
    else:
        target.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--self-test']:
        import copy
        import tempfile
        from feedback_posture_ppo import FeedbackPPO
        with tempfile.TemporaryDirectory(prefix='ppo_weight_audit_') as directory:
            folder = Path(directory)
            model = FeedbackPPO(torch.zeros(417), torch.ones(417), 9, 'cpu')
            model.save(folder/'initial.pt')
            obs = torch.randn(32, 417)
            actions, logp, _, means = model.choose(obs)
            stats = model.update(obs, actions, logp, means, actions[:, 0] + obs[:, 0])
            model.save(folder/'model_01.pt')
            (folder/'learning_audit.json').write_text(json.dumps([dict(updates=1, optimizer=stats)]))
            audit(folder, 'model_01.pt')
            initial = torch.load(folder/'initial.pt', weights_only=False)
            final = torch.load(folder/'model_01.pt', weights_only=False)
            errors = 0
            for kind in ['unchanged', 'stale_adam', 'nonfinite']:
                bad = copy.deepcopy(final)
                if kind == 'unchanged':
                    bad['actor_state_dict'] = initial['actor_state_dict']
                elif kind == 'stale_adam':
                    for state in bad['actor_optimizer']['state'].values():
                        state['step'].zero_()
                else:
                    next(iter(bad['critic_state_dict'].values())).flatten()[0] = float('nan')
                torch.save(bad, folder/'bad.pt')
                try:
                    audit(folder, 'bad.pt')
                except AssertionError:
                    errors += 1
            assert errors == 3
        print('WEIGHT_AUDITOR_UNIT_PASS: actual update accepted; unchanged weights, stale Adam and NaN rejected')
    else:
        audit(sys.argv[1], sys.argv[2])
