"""Install the aligned descent and then the two seated-posture refinements."""
import json
from pathlib import Path
import sys

if __name__=='__main__':
    original,flag,profile,launcher=sys.argv[1:5]
    assert flag=='--arm-worker'
    sys.path[:0]=[str(Path(original).parent),str(Path(launcher).parent),str(Path(launcher).parent/'trainer')]
    output=Path(sys.argv[sys.argv.index('--output')+1])
    config=json.loads((output.parent/'step_config.json').read_text())
    from controlled_seat_step_worker import install_step
    from pelvis_marker_worker import install
    from rescan_sitting import attach as rescan
    from sitting_load_effort import attach as rest
    import evaluate_armchair_pair as pair

    install_step(output,config)
    before=pair.trace_contacts

    def run(launch,options):
        def after(l,o):
            return rest(config,lambda:before(l,o))
        return rescan(config,launch,options,after)

    pair.trace_contacts=run
    install(Path(original),Path(profile),Path(launcher),sys.argv[5:])
