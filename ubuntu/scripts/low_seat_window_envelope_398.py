"""Complete nominal path with a continuous 20ms inverse-filter envelope.
For a piecewise-linear command c extended constantly outside its time domain,
U(t)=10c(t)-9c(t-.02) is piecewise linear. Its only possible breakpoints are
command-knot times and those times plus .02. Checking every breakpoint bounds
EVERY ideal shared-clock target, including ticks that straddle geometric knots.
This is an additional research envelope, not an actual float32 filter certificate.
Both the 25-sample synthetic hover hold and all mandatory contact events remain.
All original physical/model rows and inward research envelopes remain final checks.
"""
from pathlib import Path
import numpy as np
from low_seat_event_clock_v2_386 import make_event,timing,GROUPS,clock_samples,clock_rows
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')

def envelope(C,dt,lo,hi,ulp,CJ=None,dtJ=None):
    # A duplicate command at the END of the .5s synthetic hold is a real vertex.
    ids=np.r_[np.arange(100),99,np.arange(100,115)]
    durations=np.r_[dt[:99],.5,dt[99:]]
    times=np.r_[0.,np.cumsum(durations)]
    values=C[ids]
    differentiating=CJ is not None
    if differentiating:
        assert dtJ is not None
        n=CJ.shape[-1]
        TJ=np.vstack([np.zeros(n),np.cumsum(np.vstack([dtJ[:99],np.zeros(n),dtJ[99:]]),axis=0)])
        VJ=CJ[ids]
    targets=[];queries=[];query_jac=[];target_jac=[];locations=[]
    def sample(t,tj):
        if t<=0:
            return values[0],(VJ[0] if differentiating else None),(-1,0.)
        if t>=times[-1]:
            return values[-1],(VJ[-1] if differentiating else None),(115,0.)
        i=int(np.searchsorted(times,t,side='right')-1)
        a=(t-times[i])/(times[i+1]-times[i]);value=(1-a)*values[i]+a*values[i+1]
        if differentiating:
            slope=(values[i+1]-values[i])/(times[i+1]-times[i])
            jac=(1-a)*VJ[i]+a*VJ[i+1]+slope[:,None]*(tj-(1-a)*TJ[i]-a*TJ[i+1])[None,:]
        else:jac=None
        return value,jac,(i,float(a))
    for i in range(116):
        for shift in [0.,.02]:
            t=times[i]+shift
            if t>times[-1]:t=times[-1];tj=TJ[-1] if differentiating else None
            else:tj=TJ[i] if differentiating else None
            cur,cj,loc=sample(t,tj);prev,pj,ploc=sample(t-.02,tj)
            queries.append(t);targets.append(10*cur-9*prev);locations.append([*loc,*ploc])
            if differentiating:query_jac.append(tj);target_jac.append(10*cj-9*pj)
    targets=np.asarray(targets)
    raw=np.concatenate([hi[None,:]-targets,targets-lo[None,:]],axis=1)
    tightened=raw-np.r_[32*ulp,32*ulp]
    out=dict(ids=ids,times=times,values=values,queries=np.array(queries),targets=targets,locations=np.array(locations),raw=raw,tightened=tightened)
    if differentiating:
        UJ=np.asarray(target_jac);out.update(J=np.concatenate([-UJ,UJ],axis=1).reshape(-1,n),CJ=CJ,dtJ=dtJ,TJ=TJ,query_J=np.array(query_jac))
    return out

def make_window():
    d=make_event();p=d['p'];f=dict(np.load(D/'event_path_388/event_clock.npz'))
    initial=np.r_[f['parameters'],f['budgets']];shape=d['shape_free'];ulp=d['ulp']
    scale=np.r_[d['scale'][:20284],np.full(232*32,.03)]
    keep=np.r_[d['keep'][:20284],np.ones(232*32,bool)]
    last={}
    def evaluate(v,derivative=False):
        if 'v' not in last or not np.array_equal(v,last['v']):
            dt,T=timing(v[786:895],v[895:900]);x=d['seed'].copy();x[shape]=v[:786];x[1026:]=dt
            eq,raw=p['evaluate'](x);Y=p['last']['Y'].copy();C=np.vstack([p['basecmd'][0,p['moving']],Y[:,14:30]])
            tighten=d['static'].copy()
            for start in [12212,14036]:tighten[start:start+1824]=(dt[:,None]/.02*4*ulp).ravel()
            for start in [15860,17684]:tighten[start:start+1824]=(dt[:,None]*32*ulp).ravel()
            env=envelope(C,dt,p['command_lo'],p['command_hi'],ulp)
            iq=np.r_[raw-tighten,env['tightened'].ravel()]
            B=np.zeros((114,5))
            for g,(a,b) in enumerate(GROUPS):B[a:b,g]=.02*(dt[a:b]-.02)/(.02*(v[895+g]-(b-a)))
            last.clear();last.update(v=v.copy(),x=x,X=p['last']['X'].copy(),Y=Y,C=C,dt=dt,T=T,B=B,eq=eq,raw=raw,tighten=tighten,iq=iq,env=env)
        if not derivative:return last['eq'].copy(),last['iq'].copy()
        if 'E' in last:return last['E'],last['I']
        E0,I0=p['evaluate'](last['x'],True);I0=I0.copy()
        for start in [12212,14036]:
            for k in range(114):I0[start+k*16:start+(k+1)*16,1026+k]-=4*ulp/.02
        for start in [15860,17684]:
            for k in range(114):I0[start+k*16:start+(k+1)*16,1026+k]-=32*ulp
        M=np.c_[last['T'],last['B']]
        E=np.c_[E0[:,shape],E0[:,1026:]@M]
        I=np.c_[I0[:,shape],I0[:,1026:]@M]
        CJ=np.zeros((115,16,900));shape_inverse={int(a):j for j,a in enumerate(shape)}
        J=p['last']['J']
        for k in range(114):
            for j in range(9):
                if k*9+j in shape_inverse:CJ[k+1,:,shape_inverse[k*9+j]]=J[k,14:30,j]
        dtJ=np.c_[np.zeros((114,786)),M]
        env=envelope(last['C'],last['dt'],p['command_lo'],p['command_hi'],ulp,CJ,dtJ)
        I=np.vstack([I,env['J']]);last.update(E=E,I=I,env=env)
        return E,I
    return locals()
