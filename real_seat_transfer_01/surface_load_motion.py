"""Reach the seating surface with a camera-selected approach and transfer load through the pelvis."""
import json
import numpy as np
from corner_traction_motion import StanceTransfer as Baseline
from stance23_control import P

class StanceTransfer(Baseline):
    def __init__(self,picked,env,reference):
        super().__init__(picked,env,reference)
        profile=json.loads((env.out.parent/'landing_profile.json').read_text())
        self.landing={}
        cache={}
        with np.load(P/'intercept_landing_refs.npz') as data:self.donor=data['donor_desired']
        for j,(i,_,_) in self.cases.items():
            if str(i) not in profile['routes']:continue
            c=profile['routes'][str(i)]
            if 'diagnostic_rows' in profile:
                row=profile['diagnostic_rows'][j];assert row['asset_id']==i;c=row['config']
            filename=c['trajectory']
            if filename not in cache:
                with np.load(P/filename) as data:cache[filename]=(data['commands'],data['J'])
            paths,jac=cache[filename];v=c['variant']
            self.landing[j]=(i,c,paths[v],jac[v],dict(anchor=None,z=0.,xy=np.zeros(2)))
    def apply(self,env,tick,nominal,seat_load):
        out=super().apply(env,tick,nominal,seat_load)
        if tick<40:return out
        ids=self.legids.ravel().tolist()
        for j,(i,c,path,J,st) in self.landing.items():
            if i==6 and tick<466:
                if tick>=422:
                    idx=int(self.corridor_log[-1][j,1]);jac=self.cases[j][2][idx,84:].reshape(12,6)
                    u=np.clip((tick-422)/106,0.,1.);u=u*u*(3.-2.*u)
                    out[j,ids]+=out.new_tensor(u*(jac@np.array([.009,.002,.001,.018,.050,.015])))
                continue
            if i==21 and tick<188:
                out[j,ids]=out.new_tensor(self.donor[tick,ids]);continue
            base=path[tick,ids].copy()
            force=np.zeros(3)
            for name in ('pelvis_seat','left_thigh_seat','right_thigh_seat'):
                data=env.raw.scene[name].data
                force+=(data.normal_force_matrix_w.torch[j]+data.friction_force_matrix_w.torch[j]).sum((0,1)).detach().cpu().numpy()
            weight=env.robot.data.body_mass.torch[j].sum().item()*9.81
            load=force[2]/weight
            if st['anchor'] is None and tick>=c['start'] and load>c['trigger'] and np.min(self.corridor_log[-1][j,3:5])>.05:
                st['anchor']=env.previous[j,ids].cpu().numpy().copy()
            if st['anchor'] is not None:
                st['z']=np.clip(st['z']+.00155*(c['load']-load),-.006,.015)
                excess=np.maximum(0.,np.abs(force[:2])/weight-.27*max(load,0.))
                st['xy']=np.clip(st['xy']+.00055*np.sign(force[:2])*excess,-.005,.005)
                base=st['anchor']+J[:,:3]@np.r_[st['xy'],-st['z']]
            out[j,ids]=out.new_tensor(base)
        return out

