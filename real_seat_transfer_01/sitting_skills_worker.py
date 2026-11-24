"""Run the same standing-start simulator and instrument each learned handoff."""
import json
from pathlib import Path
import sys


if __name__ == '__main__':
    original, flag, profile, launcher = sys.argv[1:5]
    assert flag == '--arm-worker'
    sys.path[:0] = [str(Path(original).parent), str(Path(launcher).parent), str(Path(launcher).parent/'trainer')]
    output = Path(sys.argv[sys.argv.index('--output')+1])
    config = json.loads((output.parent/'step_config.json').read_text())
    skills = json.loads((output.parent/'skills.json').read_text())

    from controlled_seat_step_worker import install_step
    from pelvis_marker_worker import install
    from planted_descent import attach as planted
    from rescan_sitting import attach as rescan
    from sitting_load_effort import attach as rest
    from sitting_skills import ALIGN, Decisions, attach_wire_template
    import evaluate_armchair_pair as pair
    import seat_alignment

    decisions = Decisions(skills)
    correction = seat_alignment.correction
    heading = []

    def choose_alignment(prediction):
        row = len(heading)
        delta = correction(prediction)
        heading.append(prediction)
        action = decisions.choose(row, ALIGN, delta/seat_alignment.UNITS,
                                  abs(delta[0]) >= .012)
        return delta * (1., .8, 0., 1.15)[action]

    seat_alignment.correction = choose_alignment
    install_step(output, config)
    before = pair.trace_contacts

    def run(launch, opts):
        def after(l, o):
            def after_template():
                return planted(config, lambda: rest(config, lambda: decisions.attach(lambda: before(l,o))))

            return attach_wire_template(config, after_template)
        return rescan(config, launch, opts, after)

    pair.trace_contacts = run
    install(Path(original),Path(profile),Path(launcher),sys.argv[5:])
