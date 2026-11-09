"""Contact against a finite observed back, rather than its infinite plane.

Construction uses only measured depth and encoder/static collision geometry.
The evaluation mesh remains exclusively in the independent collision auditor.
"""
import json
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from original_plan import LeanPlan,SHAPES
from humanoid_sitting.shared_frame import apply

def clip(poly,equations):
    for e in equations:
        if len(poly)==0:break
        distance=poly@e[:3]+e[3]
        if np.all(distance<=0):continue
        if np.all(distance>0):return np.empty((0,3))
        result=[]
        for i,p in enumerate(poly):
            k=(i+1)%len(poly);q=poly[k];a,b=distance[i],distance[k]
            if a<=0:result.append(p)
            if (a<=0)!=(b<=0):result.append(p+a/(a-b)*(q-p))
        poly=np.asarray(result)
    return poly

class LeanGeometry(LeanPlan):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.lateral=np.cross(self.n,self.up)
        facet=ConvexHull(np.c_[self.cloud@self.up,self.cloud@self.lateral]).equations
        # Actual measured hull, original20mm boundary reserve and40mm height
        # reserve. No infinite-plane extension above the back or around its sides.
        self.clip_world=np.c_[facet[:,0,None]*self.up+facet[:,1,None]*self.lateral,facet[:,2]+.02]
        self.clip_world=np.vstack([self.clip_world,np.r_[self.up,-self.band[1]+.04],np.r_[-self.up,self.band[0]+.04]])
        fk=self.model.fk(self.q);triangles=[]
        for shape in json.loads(SHAPES.read_text())['shapes']:
            body=shape['body']
            if body not in ('torso_link','logo_link'):continue
            vertices=np.asarray(shape['vertices'])
            for ids in ConvexHull(vertices).simplices:
                # The unchanged physical scorer accepts only rear-local x<-.035.
                polygon=clip(vertices[ids],np.array([[1.,0.,0.,.035]]))
                if len(polygon)<3:continue
                polygon=apply(fk[body],polygon)
                triangles.extend([polygon[[0,j,j+1]] for j in range(1,len(polygon)-1)])
        self.rear_triangles=np.asarray(triangles)
        assert len(triangles)>0,'NO_ELIGIBLE_REAR_TORSO_GEOMETRY'
        old=super().geometry(self.start);new=self.geometry(self.start)
        self.diagnosis=dict(old_global_contact_above_observed_top_m=float(old[1]@self.up-self.band[1]),new_contact_above_observed_top_m=float(new[1]@self.up-self.band[1]),rear_triangle_count=len(triangles),input_is_camera_geometry=True,physical_criteria_changed=False)

    def geometry(self,x):
        root,fk=self.pose(x)
        if not hasattr(self,'rear_triangles'):return super().geometry(x)
        triangles=self.rear_triangles@root[:3,:3].T+root[:3,3]
        signed=triangles@self.clip_world[:,:3].T+self.clip_world[:,3]
        outside=np.any(np.min(signed,axis=1)>0,axis=1)
        inside=np.max(signed,axis=(1,2))<=0
        points=[triangles[inside].reshape(-1,3)]
        for tri in triangles[~outside&~inside]:
            polygon=clip(tri,self.clip_world)
            if len(polygon):points.append(polygon)
        points=np.concatenate(points)
        if not len(points):raise ValueError('NO_REAR_TORSO_OVER_OBSERVED_BACK')
        gaps=points@self.n+self.off;tip=points[gaps.argmin()]
        com=apply(root,np.average([apply(fk[b],c) for b,c in self.local],weights=self.mass,axis=0))
        return float(gaps.min()),tip,com,float(Rotation.from_matrix(root[:3,:3]).as_euler('xyz')[1])
