"""Select exposed pelvis material during unloaded approach, freeze at touch.

A centroid of a supporting face remains a supporting point only while that
face stays parallel to the support. Choosing the exposed material before
contact permits a genuine change of contact feature; loaded material never
slides or changes. This is geometry selection, not a contact measurement.
"""
import numpy as np
from humanoid_sitting.shared_frame import apply
from construct import FullRest as Base

class FullRest(Base):
    def __init__(self,*args,**kwargs):
        self.contact_frozen=False
        super().__init__(*args,**kwargs)
        self.preference_marker=self.marker.copy()

    def preference_point(self,state):
        return apply(state['root']@state['fk'][self.body],self.preference_marker)

    def exposed_local(self,root,fk):
        pose=root@fk[self.body]
        direction=pose[:3,:3].T@self.n
        vertices=self.mesh[self.body]
        projection=vertices@direction
        # All ties are on the same exposed face. Averaging keeps the witness
        # on that face. No other body or previously rejected state is searched.
        return vertices[projection<=projection.min()+1e-10].mean(axis=0)

    def material_point(self,root,fk,load):
        if not self.contact_frozen and load!=0.:
            raise ValueError('MATERIAL_CONTACT_MUST_BE_FROZEN_BEFORE_LOAD')
        marker=self.marker if self.contact_frozen else self.exposed_local(root,fk)
        return apply(root@fk[self.body],marker)

    def contact_local(self,state):
        return apply(np.linalg.inv(state['root']@state['fk'][self.body]),state['point'])

    def freeze_contact(self,state):
        assert not self.contact_frozen
        self.marker=self.contact_local(state).copy()
        self.contact_frozen=True
        self.states.clear();self.key=None;self.cache=None
