"""Apply the fixed camera-route transfer on fresh starts without using case IDs."""
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np


def attach_camera_routes(config, decisions, downstream):
    from chair_contact_motion import ChairContactMotion
    from rescan_sitting import match, points

    spec = json.loads(Path(decisions.config['route_spec']).read_text())
    path = config['planted_descent']
    with np.load(path['library'], allow_pickle=False) as z:
        clouds = {asset: z[asset+'_cloud'] for asset in spec['templates']}
    scale = np.asarray(spec['family_scale'])
    envelope = np.asarray(spec['envelope'])
    previous = ChairContactMotion.apply

    def apply(motion, learner, obs, tick, nominal):
        out = previous(motion, learner, obs, tick, nominal)
        if tick != 0:
            return out
        env = learner.env
        start, depth, intrinsics, joints, gravity, _, _, _, mount = env.standing_samples[-1]
        assert start == 0
        with np.load(env.out/'single_step.npz', allow_pickle=False) as step:
            standing = (step['options'] == 0) & step['verified_floor'][-50:].all(axis=(0,2))
            standing &= ~step['slip'][-50:].any(axis=(0,2)) & ~step['flags'].any(axis=(0,2))
            standing &= (step['floor_load'][-50:]>.15).all(axis=(0,2))
            standing &= (step['flat_speed'][-50:]<.02).all(axis=0)
        records = []
        for row, asset in enumerate(config['templates_by_row']):
            cloud = points(depth[row],intrinsics[row],mount,joints[row],gravity[row],env.model)
            estimate = match(clouds[asset], cloud)
            if estimate is None:
                records.append(dict(row=row, asset=asset, registration_ok=False, in_envelope=False))
                continue
            pose, error = estimate
            anchors = spec['templates'][asset]
            # Pose magnitude chooses the previously used lateral or angled
            # recipe. A signed neighbourhood separately controls the learned
            # residual, so reflected starts never inherit that permission.
            folded = pose.copy()
            folded[1:] = abs(folded[1:])
            prototypes = np.asarray([a['pose'] for a in anchors])
            prototypes[:,1:] = abs(prototypes[:,1:])
            distance = np.linalg.norm((prototypes-folded)/scale, axis=1)
            best = anchors[int(distance.argmin())]
            if best['planted']:
                path['rows'].append(row)
                path['treatments'][str(row)] = best['treatment']
            in_envelope = bool(standing[row] and best['options']
                               and np.all(abs(pose-np.asarray(best['pose']))<=envelope))
            if in_envelope and decisions.guarded:
                decisions.config['verified_rows'].append(row)
                decisions.approved[row] = best['options']
            records.append(dict(row=row, asset=asset, family=best['family'], registration_ok=True,
                                relative_pose=pose.tolist(), median_registration_error_m=error,
                                anchor_row=best['anchor_row'], standing=bool(standing[row]),
                                in_envelope=in_envelope, approved_options=best['options'] if in_envelope else {}))
        (env.out/'camera_routes.json').write_text(json.dumps(records,indent=2)+'\n')
        return out

    ChairContactMotion.apply = apply
    return downstream()


if __name__ == '__main__':
    original, flag, profile, launcher = sys.argv[1:5]
    assert flag == '--arm-worker'
    sys.path[:0] = [str(Path(original).parent),str(Path(launcher).parent),str(Path(launcher).parent/'trainer')]
    output = Path(sys.argv[sys.argv.index('--output')+1])
    config = json.loads((output.parent/'step_config.json').read_text())
    skills = json.loads((output.parent/'skills.json').read_text())

    from controlled_seat_step_worker import install_step
    from pelvis_marker_worker import install
    from planted_descent import attach as planted
    from rescan_sitting import attach as rescan
    from sitting_load_effort import attach as rest
    import evaluate_armchair_pair as pair
    import seat_alignment

    location = spec_from_file_location('verified_skills', skills['implementation'])
    verified_skills = module_from_spec(location)
    location.loader.exec_module(verified_skills)
    ALIGN, TRANSFER, REST = verified_skills.ALIGN, verified_skills.TRANSFER, verified_skills.REST

    class FreshDecisions(verified_skills.Decisions):
        def __init__(self, settings):
            super().__init__(settings)
            self.guarded = 'verified_rows' in settings
            self.approved = {}
            self.blocked = []

        def choose(self, row, phase, values, enabled=True):
            permitted = self.approved.get(row,{})
            if self.guarded and phase == REST and str(TRANSFER) in permitted:
                prior = self.records.get((row,TRANSFER))
                enabled = enabled and prior is not None and prior[1] == permitted[str(TRANSFER)]
            action = super().choose(row, phase, values, enabled)
            if action and self.guarded and phase in (TRANSFER,REST) and action != permitted.get(str(phase),0):
                features, _, _, used = self.records[row,phase]
                self.records[row,phase] = features, 0, 0., used
                self.blocked.append(dict(row=int(row), phase=int(phase), proposed=int(action)))
                return 0
            return action

        def save(self, file):
            super().save(file)
            (file.parent/'suppressed_options.json').write_text(json.dumps(self.blocked,indent=2)+'\n')

    decisions = FreshDecisions(skills)
    correction = seat_alignment.correction
    heading = []

    def align(prediction):
        row = len(heading)
        delta = correction(prediction)
        heading.append(prediction)
        action = decisions.choose(row, ALIGN, delta/seat_alignment.UNITS, abs(delta[0])>=.012)
        return delta * (1., .8, 0., 1.15)[action]

    seat_alignment.correction = align
    install_step(output, config)
    before = pair.trace_contacts

    def run(launch, options):
        def after(l,o):
            def template():
                return attach_camera_routes(config, decisions,
                    lambda: planted(config, lambda: rest(config, lambda: decisions.attach(lambda: before(l,o)))))

            return verified_skills.attach_wire_template(config, template)

        return rescan(config, launch, options, after)

    pair.trace_contacts = run
    install(Path(original),Path(profile),Path(launcher),sys.argv[5:])
