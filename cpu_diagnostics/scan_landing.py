"""Rank recorded landings from the standing rear-camera scan."""
import numpy as np

# Distances in metres, except the last two slopes (rise/run).
REGION = np.array([.045, .035, .085, .105])


def coordinates(features):
    values = np.array(features[..., :4], copy=True)
    values[..., 2] = np.tan(np.deg2rad(values[..., 2]))
    return values


def scan_distance(observed, sites):
    if not len(sites):
        return np.empty(0)
    geo = (coordinates(sites) - coordinates(observed)) / REGION
    image = np.quantile(abs(sites[:, 4:] - observed[4:]), .80, axis=1)
    return np.sqrt(np.sum(geo**2, axis=1) + (image/.12)**2)


def ranked(observed, data):
    if not np.isfinite(observed).all():
        return []
    # Preserve directly observed successful fallback neighborhoods. Failed
    # fallback observations allow routing in adjoining parts of that region.
    good = scan_distance(observed, data['fallback_clean']).min(initial=np.inf)
    bad = scan_distance(observed, data['fallback_fail']).min(initial=np.inf)
    if good < .11 or (good < .75 and good + .10 < bad):
        return []
    sites = data['landing_scans']
    delta = coordinates(observed) - coordinates(sites)
    # Relative to each donor's ankles, predict the change in seat level at
    # the first supporting patch and at the lowest points of both hip shells.
    xy = data['landing_xy']
    rise = delta[:, 1, None] - np.einsum('nij,nj->ni', xy, delta[:, 2:])
    clearance = data['landing_gaps'] - rise[:, 1:]
    edge_shift = delta[:, 0]
    distance = scan_distance(observed, sites)
    # A similar height at the ankles alone can hide a large height change
    # under the pelvis, or move the rim beyond the demonstrated contact patch.
    safe = ((abs(rise[:, 0]) <= .024) & (abs(edge_shift) <= .045)
            & (data['landing_inset'] + edge_shift >= -.005)
            & (clearance.min(axis=1) >= -.003) & (distance <= 2.6))
    cost = (distance + abs(rise[:, 0])/.009 + abs(edge_shift)/.035
            + np.maximum(.005-clearance.min(axis=1), 0.)/.004)
    path_ids = data['landing_ids']
    failures = scan_distance(observed, data['motion_failed_scans'])
    neg = np.full(len(data['ids']), np.inf)
    np.minimum.at(neg, data['motion_failed_ids'], failures)
    safe &= (neg[path_ids] > .14) & (distance < neg[path_ids]+.30)
    costs = np.full(len(data['ids']), np.inf)
    np.minimum.at(costs, path_ids[safe], cost[safe])
    return [int(k) for k in np.argsort(costs, kind='stable') if np.isfinite(costs[k])]


def select_landing(observed, data):
    options = ranked(observed, data)
    if not options:
        return -1
    path = options[0]
    if 'adverse_scans' in data:
        # Successful first contact can still be followed by a housing strike.
        # Compare whole-rollout outcomes of this particular approach. If its
        # closest evidence is adverse, leave control with the incumbent.
        adverse = data['adverse_scans'][data['adverse_ids'] == path]
        witnessed = data['supported_scans'][data['supported_ids'] == path]
        danger = scan_distance(observed, adverse).min(initial=np.inf)
        support = scan_distance(observed, witnessed).min(initial=np.inf)
        if danger <= 1. and danger < support:
            return -1
    return path
