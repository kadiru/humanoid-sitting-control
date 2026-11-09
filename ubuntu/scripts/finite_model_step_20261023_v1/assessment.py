"""Unmodified492 assessment functions; no model/solver imports."""
import numpy as np
OUTPUT=np.r_[np.arange(14),np.arange(15,51)]
SPEC={"closure_admission":{k:1e-13 for k in ["feet_m","orientation_0_1_vee","moment_BW_m","command_rad","torque_Nm","gap_m","COM_m"]}}
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

