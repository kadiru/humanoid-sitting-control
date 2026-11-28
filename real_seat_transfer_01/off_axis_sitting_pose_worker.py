"""Use live standing depth for the catalog policies' alignment observations."""
import json
from pathlib import Path
import sys

import numpy as np


def main():
    original, flag, _, launcher = sys.argv[1:5]
    assert flag == '--arm-worker'
    sys.path[:0] = [str(Path(original).parent), str(Path(launcher).parent),
                    str(Path(launcher).parent/'trainer')]

    import off_axis_sitting_worker
    import sitting_skills
    from off_axis_sitting_pose import estimate

    class CameraInputDecisions(sitting_skills.Decisions):
        def attach(self, downstream):
            from chair_contact_motion import ChairContactMotion

            with np.load(self.config['pose_template'], allow_pickle=False) as references:
                templates = {name:references[name] for name in references.files}
            previous = ChairContactMotion.apply

            def capture(motion, learner, obs, tick, nominal):
                if tick == 0:
                    env = learner.env
                    t, depth, intrinsics, joints, gravity, _, _, _, mount = env.standing_samples[-1]
                    assert t == 0
                    estimates = []
                    for row, asset in enumerate(self.config['pose_assets']):
                        values, pose, residual = estimate(templates[asset],
                            (depth[row],intrinsics[row],mount,joints[row],gravity[row]),env.model)
                        key = row, sitting_skills.ALIGN
                        placeholder = self.records.pop(key)
                        assert placeholder[1:] == (0, 0., False)
                        self.choose(row, sitting_skills.ALIGN, values, False)
                        estimates.append(dict(asset=asset, valid=pose is not None, inputs=values.tolist(),
                                              residual_m=residual, pose=None if pose is None else pose.tolist()))
                    (env.out/'camera_pose_observation.json').write_text(json.dumps(estimates, indent=2)+'\n')
                return previous(motion, learner, obs, tick, nominal)

            ChairContactMotion.apply = capture
            return super().attach(downstream)

    sitting_skills.Decisions = CameraInputDecisions
    off_axis_sitting_worker.main()


if __name__ == '__main__':
    main()
