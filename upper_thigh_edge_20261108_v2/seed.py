"""Fresh retargeting of a physically admitted demonstration, not a failed point."""
import numpy as np
from scipy.spatial.transform import Rotation

def construct(plan):
    q=np.asarray(plan.reference_q,float).copy()
    fk=plan.model.fk(q)
    root=plan.feet[0]@np.linalg.inv(fk[plan.foot_bodies[0]])
    x=np.r_[q[:23],root[:3,3],Rotation.from_matrix(plan.R0.T@root[:3,:3]).as_rotvec(),0.,0.]
    lower=np.r_[plan.lower+.031,[-.5,-.2,-.6],[-.65,-.65,-.35],0.,0.]
    upper=np.r_[plan.upper-.031,[.15,.2,.15],[.65,.65,.35],np.inf,14.4]
    for side in ('left','right'):
        i=plan.model.names.index(side+'_knee_joint');lower[i]=max(lower[i],.04)
    if np.any(x<lower) or np.any(x>upper):raise ValueError('ADMITTED_REFERENCE_OUTSIDE_CURRENT_MODEL_DOMAIN')
    state=plan.state(x,0.)
    return x,dict(source=plan.reference_provenance,construction='Admitted full20s known_old/8 final measured pose, rigidly reanchored to CURRENT left foot; all joint values unchanged. Full solver must close BOTH original feet and current finite contact surfaces.',
       full_3D_foot_residual=float(np.max(abs(state['foot_error']))),right_foot_initial_error=state['foot_error'][6:].tolist(),
       no_rejected_optimizer_state=True,previously_physical_success=True,current_model_pass=False,physical_success=False)
