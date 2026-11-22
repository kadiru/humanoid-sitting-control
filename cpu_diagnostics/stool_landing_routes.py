"""Choose a demonstrated leg motion from the observed stool rim and plane."""
import numpy as np


class StoolLandingRoutes:
    def __init__(self, settings):
        self.rules = settings['sites']
        self.sites = np.asarray([rule['scan'] for rule in self.rules])
        self.scale = np.asarray(settings['scale'])
        self.extensions = [(region, np.asarray(region['scans'])) for region in settings.get('extensions', [])]

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
        return rule if distance[index] <= rule['radius'] else None

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
