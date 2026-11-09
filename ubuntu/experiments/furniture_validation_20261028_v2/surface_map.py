"""Mesh membership, not an enclosing-box contact label.

The old 15 mm surface distance and 10 mm perimeter inset remain unchanged.
Internal triangle diagonals are not treated as physical surface boundaries.
Multi-normal contacts are reported as unvalidated rather than assigned a normal
that could conceal a bad contact. No actor imports this module.
"""
import numpy as np
import trimesh
from io_utils import ASSETS, read

DISTANCE = .015
INSET = .01

def segment_distance(points, segments):
    p = np.asarray(points, float)[:, None, :]
    a = segments[None, :, 0]; d = segments[None, :, 1]-a
    t = np.clip(((p-a)*d).sum(-1)/np.maximum((d*d).sum(-1), 1e-30), 0, 1)
    return np.linalg.norm(p-(a+t[..., None]*d), axis=-1).min(1)

class SurfaceMap:
    def __init__(self, case):
        self.case = case
        folder = ASSETS/'scenes'/case['id']
        z = np.load(folder/'mesh.npz')
        self.mesh = trimesh.Trimesh(z['vertices'], z['faces'], process=False)
        self.tri = self.mesh.triangles
        self.normals = self.mesh.face_normals
        self.region = np.asarray(read(folder/'target_region.json')['triangle_indices'], int)
        self.allowed = np.zeros(len(self.tri), bool); self.allowed[self.region] = True
        # Weld only for boundary bookkeeping; never alter collision triangles.
        patch = trimesh.Trimesh(self.tri[self.region].reshape(-1,3),
            np.arange(len(self.region)*3).reshape(-1,3), process=False)
        patch.merge_vertices(digits_vertex=7)
        edges = np.sort(patch.edges, axis=1)
        unique, counts = np.unique(edges, axis=0, return_counts=True)
        self.boundary = patch.vertices[unique[counts == 1]]
        assert len(self.boundary) and np.all(counts <= 2), 'NON_MANIFOLD_TARGET'

    def map_points(self, points):
        out = []
        for p in np.asarray(points, float).reshape(-1,3):
            if not np.isfinite(p).all():
                out.append(dict(valid=False, triangle=-1, distance=None, inset=None,
                                normal=[0.,0.,1.], reason='missing_contact_point')); continue
            near = trimesh.triangles.closest_point(self.tri, np.broadcast_to(p, (len(self.tri),3)))
            distances = np.linalg.norm(near-p, axis=1)
            j = int(np.argmin(distances)); distance = float(distances[j])
            inset = float(segment_distance(near[j:j+1],self.boundary)[0])
            tied = distances <= distance+1e-7
            # A seam is ambiguous if two equally close faces disagree on which
            # target/normal the contact belongs to. Never silently pick a winner.
            ambiguous = bool(np.any(self.allowed[tied] != self.allowed[j]) or
                np.any(np.linalg.norm(self.normals[tied]-self.normals[j],axis=1)>1e-5))
            valid = bool(distance < DISTANCE and self.allowed[j] and inset > INSET and not ambiguous)
            out.append(dict(valid=valid, triangle=j, distance=distance, inset=inset,
                normal=self.normals[j].tolist(), reason='valid' if valid else
                ('ambiguous_seam' if ambiguous else 'outside_target_or_perimeter')))
        return out

    def probe_points(self):
        centers = self.mesh.triangles_center[self.region]
        labels = self.map_points(centers)
        eligible = np.array([i for i,x in enumerate(labels) if x['valid']],int)
        assert len(eligible), ('NO_INTERIOR_PROBE_POINT',self.case['id'])
        # Fixed deterministic selection: largest interior triangle, then farthest
        # two remaining valid centroids. No simulator-driven point search.
        first = int(eligible[np.argmax(self.mesh.area_faces[self.region[eligible]])])
        chosen = [first]
        for _ in range(2):
            distance = np.linalg.norm(centers[eligible,None]-centers[chosen][None],axis=-1).min(1)
            chosen.append(int(eligible[np.argmax(distance)]))
        return [dict(point=centers[i].tolist(),normal=self.normals[self.region[i]].tolist(),
                     triangle=int(self.region[i])) for i in chosen]

