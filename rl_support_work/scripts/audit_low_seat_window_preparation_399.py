"""Independent arithmetic audit of continuous-window preparation398.
No kinematic projection or optimizer is executed. Rebuild breakpoint targets and
all7424 new derivative rows from saved endpoint/time derivatives. A piecewise-linear
U(t)=10c(t)-9c(t-.02) has breakpoints only at knots and knots+.02; check every interval
midpoint and all675 original20ms targets, without interpreting these as execution.
"""
from pathlib import Path
import numpy as np,json,hashlib,datetime,math
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'window_preparation_audit_399'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    O.mkdir(exist_ok=False);r=json.loads((D/'window_preparation_398/preparation.json').read_text());assert r['passed']
    for p,h in r['source_input_sha256'].items():assert sha(p)==h,p
    f=dict(np.load(D/'window_preparation_398/fixture.npz'));ids=list(range(100))+[99]+list(range(100,115));dt=list(f['dt'][:99])+[.5]+list(f['dt'][99:]);times=np.array([math.fsum(dt[:k]) for k in range(116)]);C=f['C'][ids];CJ=f['env_CJ'][ids];TJ=f['env_TJ'];total=times[-1]
    assert np.array_equal(ids,f['env_ids']);assert abs(times-f['env_times']).max()<1e-13
    def sample(q,dq=None):
        if q<=0:return C[0],CJ[0] if dq is not None else None
        if q>=total:return C[-1],CJ[-1] if dq is not None else None
        k=max(i for i in range(115) if times[i]<=q);width=times[k+1]-times[k];u=(q-times[k])/width
        value=C[k]+u*(C[k+1]-C[k])
        if dq is None:return value,None
        du=(dq-TJ[k])/width-(q-times[k])*(TJ[k+1]-TJ[k])/(width*width)
        deriv=CJ[k]+u*(CJ[k+1]-CJ[k])+(C[k+1]-C[k])[:,None]*du[None,:]
        return value,deriv
    targets=[];J=[]
    for k in range(116):
        for shift in (0.,.02):
            t=min(times[k]+shift,total);dtime=TJ[-1] if times[k]+shift>total else TJ[k]
            now,jn=sample(t,dtime);prev,jp=sample(t-.02,dtime);targets.append(10*now-9*prev);deriv=10*jn-9*jp;J.extend(np.vstack([-deriv,deriv]))
    targets=np.array(targets);J=np.array(J);raw=np.column_stack((f['hi']-targets,targets-f['lo']));tight=raw-np.r_[32*f['ulp'],32*f['ulp']]
    verr=float(abs(targets-f['env_targets']).max());jerr=float(abs(J-f['I'][20284:]).max());rerr=float(abs(tight.ravel()-f['ineq'][20284:]).max())
    assert verr<1e-12 and jerr<2e-8 and rerr<1e-12
    knots=sorted(set([0.,total]+[min(float(t+s),total) for t in times for s in (0.,.02)]))
    def U(t):return 10*sample(t)[0]-9*sample(t-.02)[0]
    interpolation_error=0.
    for a,b in zip(knots[:-1],knots[1:]):interpolation_error=max(interpolation_error,float(abs(U((a+b)/2)-(U(a)+U(b))/2).max()))
    assert interpolation_error<2e-12
    clock=np.array([U((k+1)*.02) for k in range(675)]);old=dict(np.load(D/'event_path_388/event_clock.npz'));clockerr=float(abs(clock-old['targets']).max());assert clockerr<1e-12
    # The extreme envelope bounds every shared-clock target for every joint.
    assert np.all(clock.min(0)>=targets.min(0)-1e-12) and np.all(clock.max(0)<=targets.max(0)+1e-12)
    dots=np.array([math.fsum(float(a)*float(b) for a,b in zip(row,f['direction'])) for row in np.vstack([f['E'],f['I']])]);doterr=float(abs(dots-f['analytic']).max());fderr=float(abs(dots-f['fd']).max());assert doterr<1e-12 and fderr<2e-6
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,breakpoint_targets=232,new_rows=7424,total_rows=27708,maximum_target_error=verr,maximum_derivative_rebuild_error=jerr,maximum_row_error=rerr,maximum_piecewise_linear_midpoint_error=interpolation_error,all675_clock_targets_reconstruction=clockerr,scalar_dot_error=doterr,full_directional_error=fderr,unchanged_derivative_bound=2e-6,actual_float32_schedule=False,intermediate_geometry_checked=False,physical_admission=False,source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),D/'window_preparation_398/preparation.json',D/'window_preparation_398/fixture.npz']})
    (O/'independent_audit.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
