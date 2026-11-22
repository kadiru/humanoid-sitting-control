"""Choose a demonstrated leg motion from the observed stool rim and plane."""
import numpy as np


class StoolLandingRoutes:
    def __init__(self, settings):
        self.rules = settings['sites']
        self.sites = np.asarray([rule['scan'] for rule in self.rules])
        self.scale = np.asarray(settings['scale'])
        self.extensions = [(region, np.asarray(region['scans'])) for region in settings.get('extensions', [])]
        self.stool = settings.get('stool_family')

    def match(self, observed):
        geometry = observed[:4]
        if not np.isfinite(geometry).all():
            return None
        # Each measured region ends before the nearest protected scan.
        for region, scans in self.extensions:
            distance = np.linalg.norm((scans[:, :4] - geometry) / self.scale, axis=1)
            if np.any(distance <= scans[:, 4]):
                return region
        distance = np.linalg.norm((self.sites - geometry) / self.scale, axis=1)
        index = int(np.argmin(distance))
        rule = self.rules[index]
        if distance[index] <= rule['radius']:
            return rule
        if self.stool is None:
            return None
        # Use the rim, cross-slope, and eight central depth tiles that see the
        # top and its two sides. Retain locally verified nominal descents.
        fixed = np.asarray(self.stool['nominal_scans'])
        if np.linalg.norm((fixed - geometry) / self.scale, axis=1).min() <= self.stool['nominal_radius']:
            return None
        indices = self.stool['depth_tiles']
        measured = np.r_[geometry, observed[4:][indices]]
        bounds = np.asarray(self.stool['observation_bounds'])
        if np.all((measured >= bounds[0]) & (measured <= bounds[1])):
            # A distant front rim leaves less seated overlap when the pelvis
            # moves forward. Preserve that overlap during the planted descent.
            reach = self.stool['rim_balance']
            travel = (geometry[0] - reach['zero_shift_edge_m']) * reach['gain']
            return dict(self.stool, forward_shift_m=float(np.clip(travel, 0., reach['max_shift_m'])))
        return None

    def install(self):
        import hold_clearance

        prior_choose = hold_clearance.choose

        def choose_landing(observed, original, library):
            previous = prior_choose(observed, original, library)
            rule = self.match(observed)
            if rule is None:
                return previous
            if 'from_motion' in rule and (previous < 0 or library['ids'][previous] != rule['from_motion']):
                return previous
            matches = np.flatnonzero(library['ids'] == rule['to_motion'])
            if len(matches) != 1:
                raise ValueError('Stool route must resolve to one frozen motion')
            return int(matches[0])

        # Selection precedes the existing hip-clearance and grip guards.
        hold_clearance.choose = choose_landing
