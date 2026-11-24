"""Insert the geometry-conditioned approach after the incumbent's motion selection."""
import sys
from pathlib import Path

if __name__ == '__main__':
    original, flag, profile, launcher = sys.argv[1:5]
    assert flag == '--arm-worker'
    sys.path[:0] = [str(Path(original).parent),str(Path(launcher).parent),str(Path(launcher).parent/'trainer')]
    import evaluate_armchair_pair as pair
    from scale_approach import attach_approach
    from pelvis_marker_worker import install
    previous = pair.trace_contacts
    pair.trace_contacts = lambda launch, opts: attach_approach(launch, opts, previous)
    install(Path(original),Path(profile),Path(launcher),sys.argv[5:])
