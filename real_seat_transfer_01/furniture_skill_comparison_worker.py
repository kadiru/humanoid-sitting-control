"""Compare separate skill policies with a shared policy on the same approach."""
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

    settings = json.loads((output.parent/'skills.json').read_text())
    policy = Decisions(settings)
    # The matched catalog placements start in a standing stance. Both methods
    # skip the optional alignment step and use the same camera-based approach.
    for row in settings['hold_rows']:
        policy.choose(row, ALIGN, np.zeros(3), False)
    inherited = pair.trace_contacts

    def with_skills(launch, options):
        return attach_approach(launch, options,
                               lambda l, o: policy.attach(lambda: inherited(l, o)))

    pair.trace_contacts = with_skills
    install(Path(original), Path(profile), Path(launcher), sys.argv[5:])


if __name__ == '__main__':
    main()
