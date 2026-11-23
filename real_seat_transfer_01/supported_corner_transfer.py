"""Retime only the measured 5 mm of seat loading, then retain the verified leg targets."""
import numpy as np
from measured_corner_motion import StanceTransfer as Parent
from stance23_control import P
class StanceTransfer(Parent):
    def __init__(self,picked,env,ref):
        super().__init__(picked,env,ref)
        p=np.load(P/'supported_corner_transfer.npz');self.seat_ids=p['joints'].tolist();self.setpoints=p['command'];self.force_idx=int(p['first_load'])
        self.capture={};self.transfer_log=[]
    def apply(self,env,t,before,seat_load):
        out=super().apply(env,t,before,seat_load);log=np.zeros((len(out),4))
        for j,(idx,_,_) in self.cases.items():
            if idx!=30 or t<572:continue
            if j not in self.capture:self.capture[j]=[0.,env.previous[j,self.seat_ids].cpu().numpy().copy()]
            phase,start=self.capture[j];load=float(seat_load[j]);q=min(int(phase),len(self.setpoints)-1)
            target=self.setpoints[q]+np.clip(start-self.setpoints[0],-.015,.015)*max(0.,1-phase/70.)
            out[j,self.seat_ids]=out.new_tensor(target)
            # Travel faster through the unloaded gap. Reduce speed at the first
            # load and follow the observed force transfer to its settled motor targets.
            pace=3.6 if phase<self.force_idx and load<.015 else 1.9
            self.capture[j][0]=min(len(self.setpoints)-1,phase+pace)
            log[j]=[phase,load,pace,t]
        self.transfer_log.append(log)
        if t==999:np.savez_compressed(env.out/'corner_load_transfer.npz',data=np.stack(self.transfer_log))
        return out
