"""Track the pelvis contact patch over the rim using joint encoders and gravity before loading."""
import numpy as np
from scipy.spatial.transform import Rotation
from protected_surface_motion import StanceTransfer as ProtectedTransfer
from stance23_control import P
class StanceTransfer(ProtectedTransfer):
    def __init__(self,picked,env,ref):
        super().__init__(picked,env,ref)
        self.targets={}
        for j,(i,c,path,J,state) in self.landing.items():
            if 'marker' not in c:continue
            with np.load(P/c['trajectory']) as z:
                v=c['variant'];self.targets[j]=(z['goals'][v],float(z['caps'][v]),dict(trim=np.zeros(12),acc=np.zeros(3),drop=0.,inside=0))
    def apply(self,env,tick,nominal,load):
        out=super().apply(env,tick,nominal,load)
        if tick<475:return out
        ids=self.legids.ravel().tolist()
        for j,(goals,floor,state) in self.targets.items():
            _,c,path,J,pressure=self.landing[j];config=c['marker']
            if pressure['anchor'] is not None:continue
            h=self.corridor_log[-1][j];k=int(h[1]);ref=self.cases[j][2][k]
            R=Rotation.from_rotvec(-h[8:11]).as_matrix()@ref[72:81].reshape(3,3)
            sensed=ref[69:72]-h[5:8]+R@np.array(config['skin'])
            goal=goals[tick].copy();error_xy=goal[:2]-sensed[:2]
            if tick>=config['cut'] and np.linalg.norm(error_xy)<.009 and min(h[3:5])>.05:state['inside']+=1
            elif not state['drop']:state['inside']=0
            if state['inside']>=3 or state['drop']>0:
                state['drop']=min(goal[2]-floor,state['drop']+.00034)
            goal[2]-=state['drop'];error=goal-sensed
            state['acc']=np.clip(state['acc']+.02*config['integral']*error,[-.029,-.024,-.016],[.029,.024,.019])
            delta=np.clip(config['gain']*error+state['acc'],[-.047,-.031,-.023],[.036,.031,.028])
            want=J[:,:3]@delta
            state['trim']+=np.clip(np.clip(want,-.18,.18)-state['trim'],-.0025,.0025)
            out[j,ids]+=out.new_tensor(state['trim'])
        return out

