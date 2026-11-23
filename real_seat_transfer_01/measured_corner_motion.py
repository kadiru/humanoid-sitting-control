"""Track a seat approach in the measured right-shoe heading, then regulate seat pressure."""
import json
import numpy as np
from scipy.spatial.transform import Rotation
from sole_centered_mount import StanceTransfer as Parent
from stance23_control import P

class StanceTransfer(Parent):
    def __init__(self,picked,env,ref):
        routes=json.loads((P/'stance23_routes.json').read_text())
        real=[routes[row[1]] if row[-1] else -1 for row in picked]
        specs=json.loads((P/'measured_corner_routes.json').read_text())
        self.cases={};sub=list(picked)
        for j,i in enumerate(real):
            if str(i) not in specs:continue
            spec=specs[str(i)]
            self.cases[j]=(i,spec,np.load(P/(f'measured_corner_path_{i}.npz' if i==30 else f'hover_edge_path_{i}.npz' if i==12 else f'contact_corridor_path_{i}.npz'))['motion'])
            if spec['step']:
                rec=list(sub[j]);rec[1]=routes.index(16);sub[j]=tuple(rec)
        super().__init__(sub,env,ref);self.traced=np.load(P/'corridor_trace_correction.npz')['delta'];self.live={};self.corridor_log=[]
        self.legids=np.array([[0,3,7,11,15,19],[1,4,8,12,16,20]])
    def apply(self,env,t,before,seat_load):
        out=super().apply(env,t,before,seat_load);d=env.robot.data
        log=np.full((len(out),17),np.nan)
        for j,(i,s,path) in self.cases.items():
            start=s['start'];end=s['end'];p=seat_load[j]
            if t<start:continue
            if j not in self.live:
                self.live[j]=dict(phase=float(start),last=None,speed=np.zeros(6),trim=np.zeros(12),press=0.,stable=0,hold=None,begin=t,anchor=env.previous[j,:23].cpu().numpy().copy())
            state=self.live[j];k=int(state['phase']);row=path[k]
            up=-d.projected_gravity_b.torch[j].cpu().numpy();q=d.joint_pos.torch[j].cpu().numpy()
            # Right sole retains its starting heading, including when the left steps.
            from scipy.spatial.transform import Rotation as Rot
            tf=np.eye(4)
            for ang,(aa,bb),axis in zip(q[self.legids[1]],self.mats[1].cpu().numpy(),self.corr_axes):
                spin=np.eye(4);spin[:3,:3]=Rot.from_rotvec(axis*ang).as_matrix();tf=tf@aa@spin@bb
            front=tf[:3,0].copy();front-=front.dot(up)*up;front/=np.linalg.norm(front)
            H=np.stack([front,np.cross(up,front),up])
            cm,feet=self.mmodel.estimate(q,H);relative=-feet.mean(0)
            desired_R=Rotation.from_matrix(row[72:81].reshape(3,3)).as_matrix()
            err=np.r_[row[69:72]-relative,Rotation.from_matrix(desired_R@H.T).as_rotvec()]
            if state['last'] is not None:state['speed']=.84*state['speed']+.16*(err-state['last'])*50
            state['last']=err
            delta=np.clip(err+.11*state['speed'],[-.037,-.027,-.027,-.115,-.13,-.12],[.037,.027,.027,.115,.13,.12])
            if k>=end-16 or p>.038:
                state['press']=np.clip(state['press']+.00040*(.18-p),-.010,.017)
                delta[2]-=state['press']
            gain=row[84:].reshape(2,6,6)
            fb=(gain@(delta*np.array([.80,.66,.65,.57,.67,.71]))).ravel()
            state['trim']+=np.clip(np.clip(fb,-.105,.105)-state['trim'],-.0022,.0022)
            ahead=path[min(end,k+6)];cmd=row[:23].copy()+.62*(ahead[:23]-row[:23])+np.clip(p/.20,0,1.1)*row[23:46]
            cmd[self.legids.ravel()]+=state['trim'];cmd+=self.traced[i,k]
            # The pilot tracked both feet up to mid-descent, then fell back.
            # Use the ankles for the net support moment. Use the proximal leg
            # joints for the pelvis task, so the two loops do not undo each other.
            bw=float(d.body_mass.torch[j].sum().item()*9.81)
            ft=np.array([env.raw.scene[n].data.normal_force_matrix_w.torch[j,:,:,2].sum().item()/bw for n in ('rest_left_floor','rest_right_floor')])
            center=cm-feet.mean(0);ec=center[:2]-row[81:83]
            old=state.get('ec',ec);rate=.77*state.get('ec_rate',np.zeros(2))+.23*(ec-old)*50
            state['ec']=ec;state['ec_rate']=rate
            cop=center[:2]+np.clip(center[2],.4,.84)/9.81*(8.2*ec+5.3*rate)
            part=np.maximum(.015,ft);part=part/part.sum();relfeet=feet[:,:2]-feet[:,:2].mean(0)
            width=relfeet[0,1]-relfeet[1,1]
            desired_share=np.clip((cop[1]-relfeet[1,1])/max(.10,width),.055,.945)
            fraction_error=desired_share-part[0]
            filtered=.85*state.get('share_err',fraction_error)+.15*fraction_error
            state['share_err']=filtered
            delta=cop-part@relfeet
            vel=d.joint_vel.torch[j,:23].cpu().numpy()
            axesbank=np.load(P/'stance23_balance_refs.npz')['axes'] if not hasattr(self,'axisbank') else self.axisbank
            self.axisbank=axesbank
            for side,ids in enumerate(self.legids):
                tf=np.eye(4);rotaxes=[];pivots=[]
                for ang,(aa,bb),axis in zip(q[ids],self.mats[side].cpu().numpy(),axesbank[side]):
                    tf=tf@aa;rotaxes.append(H@tf[:3,:3]@axis);pivots.append(H@tf[:3,3])
                    spin=np.eye(4);spin[:3,:3]=Rot.from_rotvec(axis*ang).as_matrix();tf=tf@spin@bb
                normal=ft[side]*bw
                endpos=H@tf[:3,3]
                Jv=np.cross(rotaxes,endpos-pivots)
                push=(1 if side==0 else -1)*np.clip(filtered,-.23,.23)*bw*min(1.,ft.sum())
                kp_leg=np.array([200.,150.,150.,200.])
                cmd[ids[:4]]+=np.clip(-1.45*Jv[:4,2]*push/kp_leg,-.070,.070)
                mx,my=np.clip(delta,[-.064,-.030],[.13,.030])
                tau=np.stack(rotaxes[-2:])@np.array([-normal*my,normal*mx,0.])
                aj=ids[-2:];balance=q[aj]+tau/160.+.048*vel[aj]
                cmd[aj]=.93*balance+.07*cmd[aj]
            w=min(1.,(t-state['begin']+1)/22);cmd=w*cmd+(1-w)*state['anchor']
            state['stable']=state['stable']+1 if p>.072 and ft.min()>.06 else 0
            if state['stable']>=4 and state['hold'] is None:state['hold']=env.previous[j,:23].cpu().numpy().copy()
            if state['hold'] is not None:cmd=state['hold'].copy()
            hand=getattr(env,'supported_lap',None)
            if hand is not None and bool(hand.started[j]):cmd[self.arms]=out[j,self.arms].cpu().numpy()
            out[j,:23]=out.new_tensor(cmd);state['phase']=min(end,state['phase']+(1. if np.max(np.abs(ec+.18*rate))<.027 else .45))
            log[j]=[i,k,p,*ft,*err,*ec,state['press'],float(state['hold'] is not None),*relative[:2]]
        self.corridor_log.append(log)
        if t==999:np.savez_compressed(env.out/'corridor_feedback.npz',data=np.stack(self.corridor_log))
        return out
    @property
    def corr_axes(self):
        if not hasattr(self,'cached_corr_axes'):self.cached_corr_axes=np.load(P/'stance23_balance_refs.npz')['axes'][1]
        return self.cached_corr_axes

