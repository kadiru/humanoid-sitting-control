"""Match a standing scan to a verified contact motion while retaining the live hold actor."""
import numpy as np
import torch

SCAN_TICKS = (0, 10, 20, 30, 40)
# Rim distance, top height, pitch (degrees) and transverse slope.
CAPTURE = np.array([.012, .010, 1.2, .028])


def depth_tiles(images):
    """Median range in 48 patches. Capped far returns retain the chair silhouette."""
    z = np.clip(np.nan_to_num(images, nan=1.5, posinf=1.5, neginf=1.5), .08, 1.5)
    return np.median(z.reshape(*z.shape[:-2], 6, 16, 8, 16), axis=(-3, -1)).reshape(*z.shape[:-2], 48)


def descriptor(rims, planes, tiles):
    visible = np.isfinite(rims).all(-1) & (planes[:, 4] > .68)
    geom = np.full(4, np.nan)
    if visible.sum() >= 4:
        geom = np.median(np.column_stack((rims[visible], planes[visible, 3] / planes[visible, 4])), axis=0)
    return np.r_[geom, np.median(tiles, axis=0)]


def distances(observed, examples):
    if not len(examples):
        return np.empty(0)
    # A surface without a directly observed rim needs a closer depth-image match.
    direct = np.isfinite(observed[0])
    same = np.isfinite(examples[:, 0]) == direct
    out = np.full(len(examples), np.inf)
    if same.any():
        error = abs(examples[same, 4:] - observed[4:])
        image_distance = np.quantile(error, .80, axis=1) / (.040 if direct else .012)
        if direct:
            image_distance = np.maximum(image_distance, (abs(examples[same, :4] - observed[:4]) / CAPTURE).max(1))
        out[same] = image_distance
    return out


def select_extension(observed, library):
    sites = library['sites']
    cost = distances(observed, sites) / library['site_radius']
    # Interpolate only between two measured clean executions of the same
    # motor path, inside its validated interval of directly observed planes.
    links = library['links']
    if len(links) and np.isfinite(observed[0]):
        a, b = sites[links[:,0]], sites[links[:,1]]
        span = (b[:,:4]-a[:,:4])/CAPTURE
        t = np.sum((observed[:4]-a[:,:4])/CAPTURE*span,axis=1)/np.sum(span**2,axis=1)
        interior = (t>0) & (t<1)
        if interior.any():
            expected = a[interior]+t[interior,None]*(b[interior]-a[interior])
            # Keep a narrower tolerance across the segment than at a tested site.
            cross = distances(observed,expected)/.65
            np.minimum.at(cost,links[interior,0],cross)
    j = int(cost.argmin()) if len(cost) else -1
    if j<0 or cost[j]>1.: return -1
    protected = distances(observed,library['protected']).min(initial=np.inf)
    return int(library['site_paths'][j]) if cost[j]+.10<protected else -1


def select(observed, library):
    if 'sites' in library:
        added = select_extension(observed,library)
        if added>=0: return added
    d = distances(observed, library['features'])
    if not np.isfinite(d).any():
        return -1
    j = int(d.argmin())
    keep = distances(observed, library['retain']).min(initial=np.inf)
    return j if d[j] <= 1. and d[j] + .10 < keep else -1


class ChairContactMotion:
    def __init__(self, library, count):
        self.library = library
        self.measurements = []
        self.selected = np.full(count, -1)

    def apply(self, learner, obs, tick, nominal):
        if tick in SCAN_TICKS:
            env = learner.env
            assert env.live[0] == env.standing_samples[-1][0] == tick
            rims = [[m['edge'], m['height'], m['pitch']] if m is not None and m['direct'] else [np.nan]*3
                    for m in env.live[1]]
            self.measurements.append((rims, env.geometry.copy(), depth_tiles(env.standing_samples[-1][1])))
        if tick == SCAN_TICKS[-1]:
            rims, planes, tiles = map(np.asarray, zip(*self.measurements, strict=True))
            for i in range(len(obs)):
                self.selected[i] = select(descriptor(rims[:, i], planes[:, i], tiles[:, i]), self.library)
            self.rows = np.flatnonzero(self.selected >= 0)
            self.paths = torch.as_tensor(self.library['paths'][self.selected[self.rows]], device=obs.device)
        if tick == 999:
            np.savez_compressed(learner.env.out/'chair_contact_motion.npz', selected=self.selected,
                                sources=self.library['ids'])
        if tick < 50 or not len(self.rows):
            return nominal
        phase = np.clip((tick-50) / 100., 0., 1.)
        target = self.paths[:, tick]
        if tick >= 750:
            target = target + learner.actor(learner.normalize(obs))[self.rows]
        out = nominal.clone()
        out[self.rows] = torch.lerp(nominal[self.rows], target, float(phase**2 * (3.-2.*phase)))
        return out
