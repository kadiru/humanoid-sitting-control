"""Lift the left hip slightly during Monarch's recorded pre-contact force spike."""
import json
import numpy as np
from stance23_control import P
from surface_load_motion import StanceTransfer as SurfaceTransfer
class StanceTransfer(SurfaceTransfer):
    def __init__(self,picked,env,reference):
        super().__init__(picked,env,reference)
        d=json.loads((env.out.parent/'landing_profile.json').read_text())
        self.monarch={}
        for j,(i,_,_) in self.cases.items():
            if i==30 and 'monarch' in d:
                self.monarch[j]=d['diagnostic_rows'][j]['config'] if 'diagnostic_rows' in d else d['monarch']
        with np.load(P/'intercept_landing_refs.npz') as z:self.monarch_jac=z['donor_J']
    def apply(self,env,tick,command,seat_load):
        out=super().apply(env,tick,command,seat_load)
        if tick<475 or tick>=615:return out
        ids=self.legids.ravel().tolist()
        for j,c in self.monarch.items():
            # A smooth displacement around the original left-hip contact interval.
            start,peak,stop=c['window']
            if tick<start or tick>=stop:continue
            u=min(1.,(tick-start)/(peak-start),(stop-tick)/(stop-peak))
            w=u*u*(3.-2.*u)
            if c.get('replay'):base=self.donor[tick,ids].copy()
            else:base=out[j,ids].detach().cpu().numpy().copy()
            out[j,ids]=out.new_tensor(base+w*self.monarch_jac[tick]@np.array(c['pose']))
        return out

