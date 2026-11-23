"""Use the IMU, encoders and floor loads to keep the moving body over the stance shoe."""
import numpy as np
from impact_order_checked_motion import StanceTransfer as Prior
from stance23_control import P

class StanceTransfer(Prior):
    def __init__(self,picked,env,ref):
        super().__init__(picked,env,ref)
        self.frame_bank=np.load(P/'stance_frame_perch.npz')
        self.sole_state={};self.sole_trace=[]
    def edge_mount_action(self,j,env,out,t,foot,p):
        d=env.robot.data;up=-d.projected_gravity_b.torch[j].cpu().numpy()
        q=d.joint_pos.torch[j].cpu().numpy()
        pose,_=self.stepper.foot(1,q[self.stepper.ids2[1]])
        front=pose[:3,0].copy();front-=front.dot(up)*up;front/=np.linalg.norm(front)
        H=np.stack([front,np.cross(up,front),up])
        cm,feet=self.mmodel.estimate(q,H);v=cm-feet[1]
        state=self.sole_state.setdefault(j,{'old':v,'rate':np.zeros(3),'corr':np.zeros(9),'clock':None,'drop':0.})
        rate=.80*state['rate']+.20*(v-state['old'])*50
        state['old']=v;state['rate']=rate
        cp=v[:2]+rate[:2]*np.sqrt(np.clip(v[2],.35,.85)/9.81)
        entry=self.sweep[0,23:46]
        if state['clock'] is None:
            ready=(foot[0]<.016 and foot[1]>.86 and np.max(abs(q[:23]-entry))<.060 and abs(cp[1])<.029 and -.020<cp[0]<.087)
            if not ready:return out[j,:23],-1
            state['clock']=0.;state['anchor']=env.previous[j,:23].cpu().numpy().copy();state['start']=t
        phase=int(min(state['clock'],len(self.sweep)-1));row=self.sweep[phase]
        n=np.argmin(abs(self.frame_bank['knots']-phase));features=self.frame_bank['features'][n]
        # Regulate CoM motion in the supporting shoe heading. The joints steer the
        # body, and the ankle supplies the remaining moment about the loaded sole.
        ex=v[:2]-features[:2];capture_error=ex+.23*rate[:2]
        target=self.frame_bank['gains'][n]@np.r_[-1.55*capture_error,features[2:]-up[:2]]
        correction=state['corr'];correction+=np.clip(np.clip(target,-.24,.24)-correction,-.0028,.0028)
        cmd=row[:23].copy();cmd[self.frame_bank['joints']]+=correction
        gain=float(np.clip((t-state['start'])/20,0.,1.))
        cmd=gain*cmd+(1-gain)*state['anchor']
        h=np.clip(v[2],.35,.85);cop=np.clip(v[:2]+h*(8.*ex+5.2*rate[:2])/9.81,[-.050,-.030],[.107,.030])
        force=np.clip(foot[1],.04,1.18)*float(d.body_mass.torch[j].sum().item()*9.81)
        # Project the moment into the current ankle axes as the thigh turns.
        tf,J=self.stepper.foot(1,q[self.stepper.ids2[1]])
        axis_world=H@J[3:,-2:]
        tau=axis_world.T@np.array([-force*cop[1],force*cop[0],0.])
        cmd[[16,20]]=q[[16,20]]+tau/160.
        if p>.045:
            state['drop']=min(1.,state['drop']+.004);cmd[11]-=.5*state['drop'];cmd[0]+=.23*state['drop']
        # Wait for ankle pressure and hip recovery as the predicted pressure
        # point approaches the border. Keep the contact phase responsive even then.
        speed=min(1.15,max(0.,(.031-abs(cp[1]))/.013),max(0.,(.108-cp[0])/.026),max(0.,(cp[0]+.055)/.024))
        state['clock']=min(len(self.sweep)-1,state['clock']+speed*np.clip((foot[1]-.28)/.48,0,1.))
        self.sole_trace.append([t,j,phase,*v,*cp,foot[0],foot[1],p,*correction])
        if t==999:np.savez_compressed(env.out/'stance_capture_mount.npz',records=self.sole_trace)
        return out.new_tensor(cmd),phase

