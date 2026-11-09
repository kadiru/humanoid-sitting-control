"""Bounded full-space minimax execution. Auxiliary feasibility is not actual feasibility."""
import json,time,datetime
from scipy.optimize import linprog
from low_seat_lifted_feasibility_490 import *
from low_seat_clock_transcription_444 import Mechanics,load_inputs
SPEC=dict(SPEC)
SPEC.update(executable_run_wrapper=True,max_major_iterations=24,max_actual_lps=48,max_maps=120,
    max_dyadic_trials=5,lp_time_limit_s=45.,linear_bound=1e-7,lp_tolerance=1e-9,
    acceptance_ratio=.1,minimum_trust=1/32,
    closure_admission={'feet_m':1e-13,'orientation_0_1_vee':1e-13,'moment_BW_m':1e-13,
        'command_rad':1e-13,'torque_Nm':1e-13,'gap_m':1e-13,'COM_m':1e-13},
    closure_admission_scope='Separate conservative numerical construction tests; actual signed rows and original normal/hover/kinematics gates never replaced.',
    unchanged_state_policy='Exact unchanged X inherits saved actualY/J; genuinely changed X is freshly evaluated. No root projection.',
    physical_admission=False)

def write(p,obj):p.write_text(json.dumps(obj,indent=2))
def row_values(d,f):return (d['M']@f.ravel()+d['rows']['constant'])/d['rows']['scale'],d['E']@f.ravel()+d['eq']['constant']
def assess(d,f,y):
    ar,ae=row_values(d,f);actual=np.c_[f[:,:24],y];rr,ee=row_values(d,actual)
    soft=d['rows']['keep']&d['rows']['soft'];hard=d['rows']['keep']&~soft
    err=f[:,24:]-y;h=(err[d['active']][:,OUTPUT]/d['sigma'][d['active']][:,OUTPUT]).ravel()
    z=f.ravel()[d['control_feature_rows']];box=float(min((z-d['lo']).min(),(d['hi']-z).min()))
    phase=d['rows']['groups']=='mesh_phase';fixed=bool(np.array_equal(f[d['mesh']['fixed']],d['initial'][d['mesh']['fixed']]))
    inactive=bool(np.all(f[:,16:24][d['mesh']['inactive']]==0));normal=float(abs(ee[:-1]).max());hover=float(abs(ee[-1]))
    parts={'feet_m':float(abs(err[:,[0,1,2,6,7,8]]).max()),'orientation_0_1_vee':float(abs(err[:,[3,4,5,9,10,11]]).max()),
        'moment_BW_m':float(abs(err[:,12:14]).max()),'command_rad':float(abs(err[:,15:31]).max()),
        'torque_Nm':float(abs(err[:,31:47]).max()),'gap_m':float(abs(err[:,47]).max()),'COM_m':float(abs(err[:,48:51]).max())}
    numerical_closure=all(parts[k]<=v for k,v in SPEC['closure_admission'].items())
    auxiliary_valid=bool(np.all(ar[hard]>=0) and box>=0 and fixed and inactive and np.all(f[:,16:24]>=0) and normal<=1e-13 and abs(ae[-1])<=1e-8)
    actual_gate=bool(np.all(rr>=0) and box>=0 and fixed and inactive and normal<=1e-13 and hover<=1e-8 and np.max(abs(y[:,:14]))<=1e-13 and numerical_closure)
    out=dict(phi=max(0.,float(-ar[soft].min()),float(abs(h).max())),auxiliary_soft_merit=max(0.,float(-ar[soft].min())),
        consistency_scaled_max=float(abs(h).max()),auxiliary_hard_min=float(ar[hard].min()),auxiliary_hard_pass=bool(np.all(ar[hard]>=0)),
        auxiliary_phase_min=float(ar[phase].min()),actual_all_rows_min=float(rr.min()),actual_hard_min=float(rr[hard].min()),
        actual_soft_merit=max(0.,float(-rr[soft].min())),actual_phase_min=float(rr[phase].min()),full_control_box_min=box,
        actual_normal_max=normal,actual_hover_normalized=hover,auxiliary_hover_normalized=float(abs(ae[-1])),
        actual_foot_moment_max=float(abs(y[:,:14]).max()),consistency_by_unit=parts,consistency_pass=numerical_closure,
        fixed_exact=fixed,inactive_exact=inactive,nonnegative_loads=bool(np.all(f[:,16:24]>=0)),
        auxiliary_acceptance_domain_pass=auxiliary_valid,actual_numerical_admission_gate=actual_gate,physical_admission=False)
    return out,dict(auxiliary_rows=ar,actual_rows=rr,auxiliary_equalities=ae,actual_equalities=ee,consistency=err)

def modeled_phi(d,b,t):
    soft=d['rows']['keep']&d['rows']['soft']
    return max(0.,float(-(b['r']+b['A']@t)[soft].min()),float(abs(b['h']+b['C']@t).max()))
def encoded_prediction(d,f,newf,y,J):
    delta=newf-f;modelY=y+np.einsum('nyx,nx->ny',J,delta[:,:24]);return assess(d,newf,modelY)[0]['phi']

def save_context(folder,d,f,y,J,drops,trust,b,s):
    folder.mkdir(exist_ok=False)
    np.savez_compressed(folder/'state.npz',features=f,actualY=y,J=J,r=b['r'],e=b['e'],h=b['h'],guard=b['guard'])
    for name,m in [('A',b['A']),('C',b['C']),('H',s['H'])]:sparse.save_npz(folder/(name+'.npz'),m)
    write(folder/'intervals.json',{**{k:[pair(x) for x in s[k]] for k in ['lower','upper','widths','closed_lower','closed_upper']},'owners':s['owners'],'drops':[pair(x) for x in drops],'trust':trust})

def solve(folder,d,b,s,ceiling=None):
    folder.mkdir(exist_ok=False);p=assemble(d,b,s,ceiling)
    for name in ['matrix','equality']:sparse.save_npz(folder/(name+'.npz'),p[name])
    np.savez_compressed(folder/'arrays.npz',**{k:p[k] for k in ['rhs','cost','bounds','equality_rhs']})
    write(folder/'stage.json',dict(ceiling=ceiling,primary=ceiling is None))
    start=time.monotonic()
    result=linprog(p['cost'],A_ub=p['matrix'],b_ub=p['rhs'],A_eq=p['equality'],b_eq=p['equality_rhs'],bounds=p['bounds'],method='highs',options=dict(time_limit=45.,primal_feasibility_tolerance=1e-9,dual_feasibility_tolerance=1e-9))
    meta=dict(status=int(result.status),message=str(result.message),candidate=result.x is not None,actual_lp_call=True,success=bool(result.success),iterations=int(result.nit),elapsed_s=time.monotonic()-start)
    if result.x is not None:
        t=result.x;v=max(0.,float((p['matrix']@t-p['rhs']).max()),float(abs(p['equality']@t-p['equality_rhs']).max()),float((p['bounds'][:,0]-t).max()),float((t-p['bounds'][:,1]).max()))
        meta.update(linear_violation=v,linear_pass=bool(np.all(np.isfinite(t)) and v<=1e-7),objective=float(p['cost']@t));np.save(folder/'witness.npy',t)
    write(folder/'solver.json',meta);return result.x,meta

class Evaluation:
    def __init__(self,d):self.d=d;self.inputs=load_inputs();self.counts=dict(value_requests=0,value_state_records=0,jacobian_requests=0,jacobian_state_records=0,root_updates=0,initial_lifts=0)
    def evaluate(self,oldX,newX,old,kind):
        changed=np.flatnonzero(np.any(oldX!=newX,axis=1));out=old.copy()
        if len(changed):
            oracle=Mechanics(self.inputs,self.d['mesh']['canonical'][changed]);fun=oracle.values if kind=='value' else oracle.jacobian
            out[changed]=fun(newX[changed]);self.counts[kind+'_requests']+=1;self.counts[kind+'_state_records']+=len(changed)
        return out,changed
