"""Unmodified independent599 metric implementation; no producer import."""
import numpy as np
OUT=np.r_[np.arange(14),np.arange(15,51)]
def summary(d,f,y):
    row=d['row'];soft=row['keep']&row['soft'];hard=row['keep']&~soft;active=d['z']['active'];err=f[:,24:]-y
    ar=(d['M']@f.ravel()+row['constant'])/row['scale'];rr=(d['M']@np.c_[f[:,:24],y].ravel()+row['constant'])/row['scale'];ae=d['E']@f.ravel()+d['eq']['constant'];ee=d['E']@np.c_[f[:,:24],y].ravel()+d['eq']['constant']
    h=(err[active][:,OUT]/d['z']['sigma'][active][:,OUT]).ravel();z=f.ravel()[d['z']['control_feature_rows']];box=float(min((z-d['z']['lower']).min(),(d['z']['upper']-z).min()))
    fixed=np.array_equal(f[d['mesh']['fixed']],d['z']['features'][d['mesh']['fixed']]);inactive=np.all(f[:,16:24][d['mesh']['inactive']]==0);normal=float(abs(ee[:-1]).max());hover=float(abs(ee[-1]));phase=row['groups']=='mesh_phase'
    parts={'feet_m':float(abs(err[:,[0,1,2,6,7,8]]).max()),'orientation_0_1_vee':float(abs(err[:,[3,4,5,9,10,11]]).max()),'moment_BW_m':float(abs(err[:,12:14]).max()),'command_rad':float(abs(err[:,15:31]).max()),'torque_Nm':float(abs(err[:,31:47]).max()),'gap_m':float(abs(err[:,47]).max()),'COM_m':float(abs(err[:,48:51]).max())}
    closure=all(x<=1e-13 for x in parts.values());valid=bool(np.all(ar[hard]>=0) and box>=0 and fixed and inactive and np.all(f[:,16:24]>=0) and normal<=1e-13 and abs(ae[-1])<=1e-8)
    final=bool(np.all(rr>=0) and box>=0 and fixed and inactive and normal<=1e-13 and hover<=1e-8 and np.max(abs(y[:,:14]))<=1e-13 and closure)
    s=dict(phi=max(0.,float(-ar[soft].min()),float(abs(h).max())),auxiliary_soft_merit=max(0.,float(-ar[soft].min())),consistency_scaled_max=float(abs(h).max()),auxiliary_hard_min=float(ar[hard].min()),auxiliary_hard_pass=bool(np.all(ar[hard]>=0)),auxiliary_phase_min=float(ar[phase].min()),actual_all_rows_min=float(rr.min()),actual_hard_min=float(rr[hard].min()),actual_soft_merit=max(0.,float(-rr[soft].min())),actual_phase_min=float(rr[phase].min()),full_control_box_min=box,actual_normal_max=normal,actual_hover_normalized=hover,auxiliary_hover_normalized=float(abs(ae[-1])),actual_foot_moment_max=float(abs(y[:,:14]).max()),consistency_by_unit=parts,consistency_pass=closure,fixed_exact=bool(fixed),inactive_exact=bool(inactive),nonnegative_loads=bool(np.all(f[:,16:24]>=0)),auxiliary_acceptance_domain_pass=valid,actual_numerical_admission_gate=final,physical_admission=False)
    return s,dict(auxiliary_rows=ar,actual_rows=rr,auxiliary_equalities=ae,actual_equalities=ee,consistency=err),h

