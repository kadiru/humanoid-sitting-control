"""Freeze body/tile separating witnesses from original standing and IK seed.

Every observed pixel triangle belongs to exactly one fixed 8x8-cell tile.
Per body/tile choose one of five sufficient halfspaces using only the original
seed and standing hulls. Hold that choice fixed throughout endpoint/path solve.
The original detailed pelvis row and original full-camera return audit remain.
"""
import numpy as np
from compact_camera import CompactCamera
from envelope_camera import EnvelopeCamera
from region_certificate import certificate_planes

class RegionCamera(CompactCamera):
    pelvis_row=EnvelopeCamera.pelvis_row

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        shapes=args[1]
        self.vertices={b:np.concatenate([np.asarray(s['vertices']) for s in shapes if s['body']==b]) for b in self.samples}
        self.camera_rotation=self.R0@self.T[:3,:3]
        self.camera_translation=-self.T[:3,:3].T@self.T[:3,3]
        self.certificates=None
        self.report={'frozen':False,'physical_success':False}

    def original_rows(self,root,fk,required_gap):
        return super().rows(root,fk,required_gap)

    def freeze_regions(self,standing,seed):
        if self.certificates is not None:raise ValueError('NO_REGION_RESELECTION')
        norm_cam=np.linalg.solve(self.camera_rotation,self.N.reshape(-1,3).T).T.reshape(self.N.shape)
        alpha=self.D-norm_cam@self.camera_translation
        tiles=[];pixel_cells=8;associations=[]
        for row in range(0,self.h-1,pixel_cells):
            bottom=min(row+pixel_cells,self.h-1)
            for col in range(0,self.w-1,pixel_cells):
                right=min(col+pixel_cells,self.w-1)
                bounds=[(col+.5-self.K[0,2])/self.K[0,0],(right+.5-self.K[0,2])/self.K[0,0],
                        (row+.5-self.K[1,2])/self.K[1,1],(bottom+.5-self.K[1,2])/self.K[1,1]]
                for feet in (False,True):
                    mask=self.valid[row:bottom,col:right].copy()
                    if feet:mask &= ~self.ground[row:bottom,col:right]
                    if not np.any(mask):continue
                    N=norm_cam[row:bottom,col:right][mask];a=alpha[row:bottom,col:right][mask]
                    A,b,detail=certificate_planes(bounds,N,a)
                    rr,cc,side=np.nonzero(mask)
                    source_ids=np.c_[rr+row,cc+col,side]
                    tiles.append((feet,A,b,len(associations)))
                    associations.append(dict(tile=[row,bottom,col,right],feet_exclude_ground=feet,
                        facet_indices=source_ids.tolist(),**detail))
        self.certificates={};choices=[];minimum=np.inf;raw_count=0
        for body,vertices in self.vertices.items():
            if body==self.body:continue
            feet=body in ('left_ankle_roll_link','right_ankle_roll_link')
            samples=[]
            for root,fk in (standing,seed):
                pose=root@fk[body]
                samples.append((vertices@pose[:3,:3].T+pose[:3,3])@self.camera_rotation+self.camera_translation)
            samples=np.concatenate(samples);unique={}
            for is_foot,A,b,association in tiles:
                if feet!=is_foot:continue
                margins=np.min(samples@A.T+b,axis=0);chosen=int(np.argmax(margins))
                margin=float(margins[chosen]);minimum=min(minimum,margin);raw_count+=1
                normal=A[chosen];offset=float(b[chosen]);key=normal.tobytes()
                # For bit-identical normals the smallest offset is strongest.
                if key not in unique or offset<unique[key][1]:unique[key]=(normal,offset)
                choices.append(dict(body=body,tile_association=association,halfspace=chosen,original_pose_minimum_m=margin))
            if unique:
                A=np.array([v[0] for v in unique.values()]);b=np.array([v[1] for v in unique.values()])
                self.certificates[body]=(A@self.camera_rotation.T,b+A@self.camera_translation)
        self.report=dict(frozen=True,source='Original current depth, standing pose and separately constructed original-input IK seed only',
            pixel_cells_per_tile=pixel_cells,facet_associations=associations,body_witness_choices=choices,
            raw_body_tile_constraints=raw_count,retained_plane_rows=sum(len(A) for A,b in self.certificates.values()),
            original_poses_minimum_certificate_m=float(minimum),original_pelvis_rule_retained=True,
            physics_steps=0,physical_success=False,unknown_pixels='No free-space assertion; full actual-mesh gate remains mandatory')

    def rows(self,root,fk,required_gap):
        if self.certificates is None:return self.original_rows(root,fk,required_gap)
        out=[]
        for body,(A,b) in self.certificates.items():
            pose=root@fk[body];P=self.vertices[body]@pose[:3,:3].T+pose[:3,3]
            out.extend(np.min(P@A.T+b,axis=0))
        out.append(self.pelvis_row(root@fk[self.body],required_gap)[0])
        return np.asarray(out)

    def jacobian(self,jets,required_gap):
        if self.certificates is None:raise ValueError('CAMERA_REGIONS_NOT_FROZEN')
        from analytic_model import point
        out=[]
        for body,(A,b) in self.certificates.items():
            pose=jets['world'][body];J=jets['Jworld'][body];V=self.vertices[body]
            indices=np.argmin((V@pose[:3,:3].T+pose[:3,3])@A.T+b,axis=0)
            derivative=np.einsum('ijk,rj->rik',J[:3,:3],V[indices])+J[:3,3]
            out.extend(np.einsum('ri,rik->rk',A,derivative))
        _,witness=self.pelvis_row(jets['world'][self.body],required_gap)
        if witness is None:out.append(np.zeros(31))
        else:
            material,normal=witness;_,derivative=point(jets['world'][self.body],jets['Jworld'][self.body],material)
            out.append(normal@derivative)
        return np.asarray(out)
