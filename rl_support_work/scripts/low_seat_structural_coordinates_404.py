"""New complete-path coordinates; offline construction only, no runtime authority.

Positive corner loads determine nominal COM before the fixed endpoint. Endpoint
loads use an exponential family with the original three equilibrium moments.
Orientation and pre-hover descent use positive cumulative increments. Five normal
approach nodes solve their declared gap together with the original foot/COM
equations, so hover and near-plane descent are construction equations. Post-hover
descent phase order and all remaining original rows are STILL explicit checks.

No failed solution is clipped. Saved379 supplies research initialization only;
the new positive loads/phases and gap equations construct a different whole path.
All original and inward constraints, the continuous-window envelope, integer event
budgets and subsequent exact float32/intermediate-geometry gates remain required.
"""
from pathlib import Path
import numpy as np
from low_seat_window_envelope_398 import make_window
from low_seat_event_clock_v2_386 import timing,GROUPS
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
FLOOR=16*float(np.finfo(np.float32).eps)

def simplex(z):
    a=np.r_[z,0.];a=np.exp(a-a.max());a/=a.sum()
    return a,(np.diag(a)-np.outer(a,a))[:,:-1]

def fixed_moments(eta,N,A,target):
    """Bounded root solve, not a path optimizer; positive loads, exact moments."""
    lam=np.array([np.log((target[0]-A.shape[1]*FLOOR)/A.shape[1]),0.,0.])
    for it in range(30):
        exponent=N@eta+A.T@lam
        if np.max(exponent)>700:raise RuntimeError('endpoint exponential domain')
        e=np.exp(exponent);w=FLOOR+e;r=A@w-target
        if abs(r).max()<=1e-13:
            H=(A*e)@A.T
            dw=e[:,None]*(N-A.T@np.linalg.solve(H,(A*e)@N))
            return w,dw,lam,it
        step=np.linalg.solve((A*e)@A.T,-r);norm=np.linalg.norm(r)
        for half in range(12):
            trial=lam+step*2.**(-half);ee=N@eta+A.T@trial
            if ee.max()<700 and np.linalg.norm(A@(FLOOR+np.exp(ee))-target)<norm:
                lam=trial;break
        else:raise RuntimeError('endpoint moment root descent domain')
    raise RuntimeError('endpoint moment root iteration cap')

def make_structural():
    d=make_window();p=d['p'];old=dict(np.load(D/'discrete_path_379/final.npz'))
    seedX=old['full_coordinates'];seedY=old['local_outputs'];shape=d['shape']
    # New structural floors are inward construction choices, not physical allowances.
    active=[];slices=[];initial=[];moment_A=[];moment_target=[];off=0
    for i in range(114):
        mask=np.any(p['null'][i]!=0,axis=1)|np.any(p['wcom'][i]!=0,axis=1)
        aa=np.flatnonzero(mask);active.append(aa);P=p['PP'][i,aa,:2]
        A=np.vstack([np.ones(len(aa)),((P-P.mean(0))/.1).T]);moment_A.append(A)
        moment_target.append(A@p['wbase'][i,aa])
        w=seedY[i,46+aa];assert np.all(w>=0)
        if i<103:
            # Fresh positive research initialization, not an admission correction.
            z=np.log((w[:-1]+FLOOR)/(w[-1]+FLOOR))
        else:z=p['null'][i,aa,:p['nullsizes'][i]].T@np.log(w+FLOOR)
        slices.append(slice(off,off+len(z)));initial.extend(z);off+=len(z)
    force_stop=off
    ori=np.diff(np.r_[0.,seedX[:103,3],1.]);assert np.all(ori>0)
    orientation_slice=slice(off,off+103);initial.extend(np.log(ori[:-1]/ori[-1]));off+=103
    pre=np.diff(np.r_[0.,seedX[:99,2]]);assert np.all(pre>0)
    descent_slice=slice(off,off+98);initial.extend(np.log(pre[:-1]/pre[-1]));off+=98
    timing_slice=slice(off,off+109);initial.extend(d['initial'][786:895]);off+=109
    budget_slice=slice(off,off+5);initial.extend(d['initial'][895:900]);off+=5
    initial=np.asarray(initial);nv=off;assert nv==895 and force_stop==580
    gap_ids=np.arange(98,103);gap_dep=np.r_[p['dependent'],2]
    last={}

    def construct(z):
        vv=np.c_[seedX[:,2:4],seedX[:,21:23],seedX[:,16:21]].copy()
        loads=np.zeros((114,8));loadJ=np.zeros((114,8,nv));G=np.zeros((900,nv))
        fullmap=np.zeros((1026,nv));moment_lam=np.zeros((11,3));moment_steps=[]
        for i,aa in enumerate(active):
            sl=slices[i];F=p['wbase'][i,aa].sum()
            if i<103:
                sw,SJ=simplex(z[sl]);w=FLOOR+(F-len(aa)*FLOOR)*sw
                wJ=(F-len(aa)*FLOOR)*SJ
                dc=(w-p['wbase'][i,aa])@p['PP'][i,aa,:2]/.1
                dcJ=p['PP'][i,aa,:2].T@wJ/.1
                vv[i,2:4]=dc;fullmap[i*9+2:i*9+4,sl]=dcJ
            else:
                N=p['null'][i,aa,:p['nullsizes'][i]]
                w,wJ,lam,steps=fixed_moments(z[sl],N,moment_A[i],moment_target[i])
                moment_lam[i-103]=lam;moment_steps.append(steps)
                dc=np.zeros(2);dcJ=np.zeros((2,len(z[sl])));vv[i,2:4]=0.
            N=p['null'][i,aa,:p['nullsizes'][i]]
            vv[i,4:]=0.;vv[i,4:4+N.shape[1]]=N.T@(w-p['wbase'][i,aa]-p['wcom'][i,aa]@dc)
            fullmap[i*9+4:i*9+4+N.shape[1],sl]=N.T@(wJ-p['wcom'][i,aa]@dcJ)
            loads[i,aa]=w;loadJ[i,aa,sl]=wJ
        ow,oJ=simplex(z[orientation_slice]);ov=FLOOR+(1-104*FLOOR)*ow
        vv[:103,1]=np.cumsum(ov)[:103]
        for i in range(103):fullmap[i*9+1,orientation_slice]=(1-104*FLOOR)*oJ[:i+1].sum(0)
        dt,T=timing(z[timing_slice],z[budget_slice]);B=np.zeros((114,5))
        for g,(a,b) in enumerate(GROUPS):B[a:b,g]=.02*(dt[a:b]-.02)/(.02*(z[budget_slice][g]-(b-a)))
        dtJ=np.zeros((114,nv));dtJ[:,timing_slice]=T;dtJ[:,budget_slice]=B
        # A modeled gap ramp, not a timer release or a substitute for current quiet.
        terminal_gap=float(p['frozen'][104]['pelvis_gap_m'])
        local_dt=dt[99:104];total=local_dt.sum();elapsed=np.r_[0.,np.cumsum(local_dt)[:4]]
        targets=.012+(terminal_gap-.012)*elapsed/total
        elapsedJ=np.vstack([np.zeros(nv),np.cumsum(dtJ[99:104],axis=0)[:4]])
        targetJ=(terminal_gap-.012)*(elapsedJ*total-elapsed[:,None]*dtJ[99:104].sum(0))/total**2
        XX=seedX.copy();XX[:,3]=vv[:,1];XX[:,21:23]=vv[:,2:4];XX[:,16:21]=vv[:,4:]
        for iteration in range(10):
            Y=p['calc'](XX);r=np.c_[Y[gap_ids,:14],Y[gap_ids,54]-targets]
            bad=abs(r).max(1)>1e-13
            if not bad.any():break
            J=p['local_jac'](XX,gap_dep)[gap_ids];K=np.concatenate([J[:,:14],J[:,54:55]],axis=1)
            step=np.linalg.solve(K,-r[...,None])[...,0];accepted=~bad;norm=np.linalg.norm(r,axis=1)
            for half in range(9):
                ids=np.flatnonzero(~accepted);trial=XX.copy();trial[gap_ids[ids,None],gap_dep]+=step[ids]*2.**(-half)
                yy=p['calc'](trial);rr=np.c_[yy[gap_ids,:14],yy[gap_ids,54]-targets]
                take=(~accepted)&(np.linalg.norm(rr,axis=1)<norm)
                XX[gap_ids[take]]=trial[gap_ids[take]];accepted|=take
                if accepted.all():break
            if not accepted.all():raise RuntimeError('five-node gap root descent domain')
        else:raise RuntimeError('five-node gap root iteration cap')
        vv[gap_ids,0]=XX[gap_ids,2];fH=vv[98,0]
        if not 99*FLOOR<fH<1:raise RuntimeError('prehover cumulative phase domain')
        fw,fJ=simplex(z[descent_slice]);increments=FLOOR+(fH-99*FLOOR)*fw
        vv[:98,0]=np.cumsum(increments)[:98]
        # All original coordinates are retained for independent complete-path checks.
        v=np.r_[vv.ravel()[shape],z[timing_slice],z[budget_slice]]
        return dict(v=v,vv=vv,loads=loads,loadJ=loadJ,fullmap=fullmap,dt=dt,dtJ=dtJ,targets=targets,targetJ=targetJ,terminal_gap=terminal_gap,orientation_increments=ov,prehover_increments=increments,fw=fw,fJ=fJ,moment_lam=moment_lam,moment_steps=np.array(moment_steps),gap_root_iterations=iteration,gap_root_X=XX)

    def evaluate(z,derivative=False):
        if 'z' not in last or not np.array_equal(z,last['z']):
            s=construct(z);eq,iq=d['evaluate'](s['v']);last.clear();last.update(s)
            last.update(z=z.copy(),eq=eq,iq=iq,X=d['last']['X'].copy(),Y=d['last']['Y'].copy(),x=d['last']['x'].copy(),C=d['last']['C'].copy(),env=d['last']['env'])
        if not derivative:return last['eq'].copy(),last['iq'].copy()
        if 'E' in last:return last['E'],last['I']
        E0,I0=d['evaluate'](last['v'],True);J=p['last']['J'];FM=last['fullmap'].copy()
        denominators=[]
        for pos,k in enumerate(gap_ids):
            g=J[k,54];denominators.append(g[0])
            if abs(g[0])<1e-6:raise RuntimeError('gap/phase implicit derivative singular')
            FM[k*9]=(last['targetJ'][pos]-g[1:]@FM[k*9+1:k*9+9])/g[0]
        fHJ=FM[98*9].copy();fH=last['vv'][98,0]
        for k in range(98):
            FM[k*9]=last['fw'][:k+1].sum()*fHJ
            FM[k*9,descent_slice]+=(fH-99*FLOOR)*last['fJ'][:k+1].sum(0)
        G=np.zeros((900,nv));G[:786]=FM[shape]
        G[786:895,timing_slice]=np.eye(109);G[895:900,budget_slice]=np.eye(5)
        E=E0@G;I=I0@G
        last.update(E=E,I=I,G=G,fullmap_final=FM,old_E=E0,old_I=I0,old_J=J.copy(),gap_derivative_denominators=np.array(denominators))
        return E,I
    return locals()
