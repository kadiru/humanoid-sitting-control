"""Independent structural-map audit404; no root/projection/path optimization.
Rebuild positive loads and derivatives, moment equations, cumulative phases, gap
targets and their implicit derivatives from saved states. The original derivative
bound is retained; arithmetic pass is not path, filter or physical admission.
"""
from pathlib import Path
import numpy as np,json,hashlib,datetime,math
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'structural_preparation_audit_v2_405'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def sm(z):
    zz=list(z)+[0.];m=max(zz);e=[math.exp(float(x)-m) for x in zz];w=np.array([x/math.fsum(e) for x in e])
    J=np.array([[w[i]*((1 if i==j else 0)-w[j]) for j in range(len(z))] for i in range(len(w))]);return w,J
def skew(a):return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
def main():
    O.mkdir(exist_ok=False);r=json.loads((D/'structural_preparation_404/preparation.json').read_text());assert r['passed']
    for p,h in r['source_input_sha256'].items():assert sha(p)==h,p
    f=dict(np.load(D/'structural_preparation_404/fixture.npz'));z=f['z'];floor=2.**(-19);assert floor==float(f['floor'])
    slices=[slice(int(a),int(b)) for a,b in f['coordinate_slices']];nv=len(z);FM=np.zeros((1026,nv));W=np.zeros((114,8));WJ=np.zeros((114,8,nv));vv=f['vv'].copy();moment_error=0.;equilibrium_derivative=0.
    for i in range(114):
        aa=np.flatnonzero(f['active'][i]);sl=slices[i];P=f['PP'][i,aa,:2];N=f['null'][i,aa];N=N[:,np.any(N!=0,axis=0)]
        F=math.fsum(map(float,f['wbase'][i,aa]))
        if i<103:
            w,J=sm(z[sl]);w=floor+(F-len(aa)*floor)*w;J=(F-len(aa)*floor)*J
            dc=(w-f['wbase'][i,aa])@P/.1;dcJ=P.T@J/.1
        else:
            A=np.vstack([np.ones(len(aa)),((P-P.mean(0))/.1).T]);target=A@f['wbase'][i,aa]
            e=np.exp(N@z[sl]+A.T@f['moment_lam'][i-103]);w=floor+e
            # Orthogonal projection in sqrt(weight) coordinates, independent of the root solver.
            Q=np.sqrt(e)[:,None]*A.T;Z=np.sqrt(e)[:,None]*N
            J=np.sqrt(e)[:,None]*(Z-Q@np.linalg.lstsq(Q,Z,rcond=None)[0])
            moment_error=max(moment_error,float(abs(A@w-target).max()));equilibrium_derivative=max(equilibrium_derivative,float(abs(A@J).max()))
            dc=np.zeros(2);dcJ=np.zeros((2,J.shape[1]))
        assert np.all(w>floor)
        W[i,aa]=w;WJ[i,aa,sl]=J;vv[i,2:4]=dc;vv[i,4:]=0.
        vv[i,4:4+N.shape[1]]=N.T@(w-f['wbase'][i,aa]-f['wcom'][i,aa]@dc)
        FM[i*9+2:i*9+4,sl]=dcJ;FM[i*9+4:i*9+4+N.shape[1],sl]=N.T@(J-f['wcom'][i,aa]@dcJ)
    ori,oJ=sm(z[slices[114]]);inc=floor+(1-104*floor)*ori;vv[:103,1]=np.cumsum(inc)[:103]
    for i in range(103):FM[i*9+1,slices[114]]=(1-104*floor)*oJ[:i+1].sum(0)
    pre,preJ=sm(z[slices[115]]);budgets=z[slices[117]];logits=z[slices[116]];groups=[(0,16),(16,64),(64,99),(99,104),(104,114)]
    dt=[];dtJ=np.zeros((114,nv));off=0
    for g,(a,b) in enumerate(groups):
        n=b-a;w,J=sm(logits[off:off+n-1]);excess=.02*(budgets[g]-n)
        dt.extend(.02+excess*w);dtJ[a:b,slices[116].start+off:slices[116].start+off+n-1]=excess*J;dtJ[a:b,slices[117].start+g]=.02*w;off+=n-1
    terminal_gap=json.loads((D/'clearance_lift_path_330/results.json').read_text())['dense']['path'][104]['pelvis_gap_m']
    dt=np.array(dt);total=math.fsum(map(float,dt[99:104]));elapsed=np.array([math.fsum(map(float,dt[99:j])) for j in range(99,104)])
    target=.012+(float(terminal_gap)-.012)*elapsed/total
    tJ=np.array([(float(terminal_gap)-.012)*(dtJ[99:j].sum(0)/total-elapsed[j-99]*dtJ[99:104].sum(0)/total**2) for j in range(99,104)])
    # Independent body-twist kinematics for the five15-coordinate root equations.
    m=BodyModel(D/'cooked_model_50/model.json');ine=json.loads((D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json').read_text());base=json.loads((D/'clearance_lift_path_330/results.json').read_text());old=base['dense']['path'];sides=['left','right']
    legs=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']];arms=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['shoulder_pitch','elbow_pitch']];q0=np.array(old[0]['q']);mass=np.array(ine['mass']);R0=Rotation.from_quat(old[0]['root'][3:]).as_matrix();end=base['endpoint'];axis=Rotation.from_matrix(R0.T@Rotation.from_quat(end['root'][3:]).as_matrix()).as_rotvec();dz=end['root'][2]-old[0]['root'][2];joints={j['name']:j for j in m.joints};parent={j['body1'][0]:j['body0'][0] for j in m.joints if j['body0']}
    # Derive plane from measured input through the same public measurement contract,
    # without calling the construction or any root solver.
    from low_seat_joint_path_controller_17 import measured
    tr=np.load(D/'normal_landing3s_06/candidate/trace.npz');ss=np.load(D/'normal_landing3s_06/candidate/sensor_inputs.npz')['sensors'];at=int(np.argmin(abs(tr['t']-1)));cal=json.loads((D.parent/'contact_effort_repair_01/plane_calibration.json').read_text());normal,plane_d,*_=measured(m,q0,ss[0,10,2,339:342],tr['geometry'][at,2],[tr['estimated_root_x'][at,2],0,0],cal)
    def below(body,ancestor):
        while True:
            if body==ancestor:return True
            if body not in parent:return False
            body=parent[body]
    max_root=0.;max_gap=0.;max_phase_derivative=0.;denoms=[]
    for pos,i in enumerate(range(98,103)):
        k=i+1;X=f['X'][i];e=old[k];u=e['u'];blend=1-(u*u*(3-2*u))
        R=Rotation.from_rotvec([base['dense']['parameters'][2]*blend,0,0]).as_matrix()@R0@Rotation.from_rotvec(axis*X[3]).as_matrix();root=np.r_[X[:2],old[0]['root'][2]+dz*X[2]];T=transform(R,root);q=q0.copy();q[legs]=X[4:16];q[arms]=q0[arms]-.35*X[2];fk=m.fk(q)
        ot=transform(Rotation.from_quat(e['root'][3:]).as_matrix(),np.array(e['root'][:3]));of=m.fk(np.array(e['q']));points={b:apply(T@fk[b],cc) for b,cc in zip(ine['body_names'],ine['local_com'])};C=sum(mm*points[b] for mm,b in zip(mass,ine['body_names']))/mass.sum();pv=apply(T,m.vertices['pelvis']);minimum=int(np.argmin(pv@normal+plane_d));gap=float(pv[minimum]@normal+plane_d);max_gap=max(max_gap,abs(gap-target[pos]));geo={}
        residual=[]
        for side in sides:
            F=T@fk[side+'_ankle_roll_link'];F0=ot@of[side+'_ankle_roll_link'];A=F0[:3,:3].T@F[:3,:3];residual.extend(F[:3,3]-F0[:3,3]);residual.extend(.1*np.array([A[2,1]-A[1,2],A[0,2]-A[2,0],A[1,0]-A[0,1]]))
        residual.extend(C[:2]-np.array(e['com'][:2])-.1*X[21:23]);max_root=max(max_root,float(abs(np.asarray(residual)).max()))
        for j in legs+arms:
            name=m.names[j];jj=joints[name];Jw=T@fk[jj['body0'][0]]@m.frames[name][0];geo[j]=(Jw[:3,'XYZ'.index(jj['axis'])],Jw[:3,3],jj['body1'][0])
        J=np.zeros((15,18))
        for col in range(16):
            rootv=np.eye(3)[:,col] if col<2 else np.array([0,0,dz]) if col==14 else np.zeros(3);rotv=R@axis if col==15 else np.zeros(3);active=[(legs[col-2],1.)] if 2<=col<14 else [(j,-.35) for j in arms] if col==14 else []
            def velocity(body,P):
                v=rootv+np.cross(rotv,P-root);w=rotv.copy()
                for j,weight in active:
                    ax,origin,child=geo[j]
                    if below(body,child):v+=weight*np.cross(ax,P-origin);w+=weight*ax
                return v,w
            for s,side in enumerate(sides):
                body=side+'_ankle_roll_link';F=T@fk[body];v,w=velocity(body,F[:3,3]);J[s*6:s*6+3,col]=v;dR=(ot@of[body])[:3,:3].T@skew(w)@F[:3,:3];J[s*6+3:s*6+6,col]=.1*np.array([dR[2,1]-dR[1,2],dR[0,2]-dR[2,0],dR[1,0]-dR[0,1]])
            J[12:14,col]=(sum(mm*velocity(b,points[b])[0] for mm,b in zip(mass,ine['body_names']))/mass.sum())[:2];J[14,col]=normal@(rootv+np.cross(rotv,pv[minimum]-root))
        J[12:14,16:18]=-.1*np.eye(2)
        rhs=-(J[:,15:18]@FM[i*9+1:i*9+4]);rhs[14]+=tJ[pos]
        solution=np.linalg.solve(J[:,:15],rhs);FM[i*9]=solution[14]
        max_phase_derivative=max(max_phase_derivative,float(abs(FM[i*9]-f['fullmap_final'][i*9]).max()))
    fH=f['vv'][98,0];fHJ=FM[98*9].copy()
    for i in range(98):
        vv[i,0]=(i+1)*floor+(fH-99*floor)*pre[:i+1].sum();FM[i*9]=pre[:i+1].sum()*fHJ;FM[i*9,slices[115]]+=(fH-99*floor)*preJ[:i+1].sum(0)
    G=np.zeros((900,nv));G[:786]=FM[f['shape']];G[786:895,slices[116]]=np.eye(109);G[895:900,slices[117]]=np.eye(5)
    load_error=float(abs(W-f['loads']).max());load_derivative=float(abs(WJ-f['loadJ']).max());coordinate_error=float(abs(vv-f['vv']).max());map_error=float(abs(G-f['G']).max());chain_error=float(max(abs(f['old_E']@G-f['E']).max(),abs(f['old_I']@G-f['I']).max()))
    direction=f['direction'];fd_error=float(abs(np.r_[f['old_E']@G@direction,f['old_I']@G@direction]-f['fd']).max())
    assert max(moment_error,equilibrium_derivative,load_error,load_derivative,coordinate_error,max_root,max_gap)<1e-10
    assert max(map_error,chain_error,max_phase_derivative)<2e-6 and fd_error<2e-6
    assert np.all(W[f['active']]>0) and np.all(W[~f['active']]==0)
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,variables=nv,positive_load_nodes=114,endpoint_moment_nodes=11,analytic_gap_root_records=5,analytic_gap_columns=90,all_rows=27708,maximum_errors=dict(moment=moment_error,moment_derivative=equilibrium_derivative,loads=load_error,load_derivative=load_derivative,coordinates=coordinate_error,root_equations=max_root,gap=max_gap,implicit_phase_derivative=max_phase_derivative,full_map=map_error,chain_rule=chain_error,directional=fd_error),unchanged_derivative_bound=2e-6,posthover_phase_order_still_explicit=True,full_geometry_checked=False,actual_float32_schedule=False,physics_ticks=0,policy_optimizer_updates=0,physical_admission=False,source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),D/'structural_preparation_404/preparation.json',D/'structural_preparation_404/fixture.npz']})
    (O/'independent_audit.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
