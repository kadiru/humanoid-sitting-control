"""Run the catalog controllers with read-only, per-decision diagnostics."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np


def main():
    original, flag, profile, launcher = sys.argv[1:5]
    assert flag == '--arm-worker'
    sys.path[:0] = [str(Path(original).parent), str(Path(launcher).parent),
                    str(Path(launcher).parent/'trainer')]
    output = Path(sys.argv[sys.argv.index('--output')+1])

    import evaluate_armchair_pair as pair
    from pelvis_marker_worker import install
    from scale_approach import attach_approach
    from sitting_skills import ALIGN, Decisions

    class ObservedDecisions(Decisions):
        def __init__(self, config):
            super().__init__(config)
            self.evaluations = []
            self.decisions = []
            self.tick = -1
            self.network.register_forward_hook(self.observe)

        def observe(self, module, inputs, logits):
            self.evaluations.append(logits[0].detach().cpu().numpy().copy())

        def choose(self, row, phase, values, enabled=True):
            before = len(self.evaluations)
            action = super().choose(row, phase, values, enabled)
            if len(self.evaluations) != before:
                assert len(self.evaluations) == before + 1
                assert int(self.evaluations[-1].argmax()) == action
                self.decisions.append((row, phase, self.tick, action))
            return action

        def save(self, path):
            super().save(path)
            calls = np.asarray(self.decisions, dtype=np.int32).reshape(-1, 4)
            logits = np.asarray(self.evaluations, dtype=np.float32).reshape(-1, 4)
            assert len(calls) == len(logits)
            np.savez_compressed(path.with_name('live_skill_logits.npz'), calls=calls, logits=logits)
            weights = Path(self.config['weights'])
            digest = hashlib.sha256(weights.read_bytes()).hexdigest()
            receipt = dict(weights=str(weights), checkpoint_sha256=digest,
                           separate_heads=self.network.separate, calls=len(calls), alignment_enabled=False)
            path.with_name('actor_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')

    settings = json.loads((output.parent/'skills.json').read_text())
    policy = ObservedDecisions(settings)
    for row in settings['hold_rows']:
        policy.choose(row, ALIGN, np.zeros(3), False)
    inherited = pair.trace_contacts

    def attach_clock(launch, options):
        from chair_contact_motion import ChairContactMotion

        original_apply = ChairContactMotion.apply

        def apply(motion, learner, obs, tick, nominal):
            policy.tick = tick
            return original_apply(motion, learner, obs, tick, nominal)

        ChairContactMotion.apply = apply
        return inherited(launch, options)

    def with_skills(launch, options):
        return attach_approach(launch, options,
                               lambda l, o: policy.attach(lambda: attach_clock(l, o)))

    pair.trace_contacts = with_skills
    install(Path(original), Path(profile), Path(launcher), sys.argv[5:])


if __name__ == '__main__':
    main()
