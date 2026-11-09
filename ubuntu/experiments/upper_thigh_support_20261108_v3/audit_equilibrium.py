"""Independent four-contact wrench closure fixtures, without robot data."""
import json
from pathlib import Path
import numpy as np

def audit():
    worst=0.;cases=0;mean_point_counterexample=0.
    W=340.;floor=-.75;yl=.1;yr=-.1
    for load in [0.,.05,.1,.2]:
        for angle in [0.,.11,-.13]:
            P=np.array([[-.2,.09,-.16],[-.19,-.09,-.15]])
            normals=np.array([[1.,angle,-.17],[1.,-angle*.7,-.21]])
            normals/=np.linalg.norm(normals,axis=1)[:,None]
            F=W*load*.5*normals;com=np.array([.014,-.003,-.04])
            G=np.array([0.,0.,W])-F.sum(0)
            M=np.cross(com,[0.,0.,W])-np.cross(P,F).sum(0)
            cop=np.array([(floor*G[0]-M[1])/G[2],(M[0]+floor*G[1])/G[2],floor])
            fz=(cop[1]-yr)*G[2]/(yl-yr);fx=(cop[0]*G[1]-M[2]-yr*G[0])/(yl-yr)
            L=np.array([fx,G[1]*fz/G[2],fz]);R=G-L
            positions=np.vstack([[cop[0],yl,floor],[cop[0],yr,floor],P])
            forces=np.vstack([L,R,F])
            residual=np.r_[forces.sum(0)-[0,0,W],np.cross(positions,forces).sum(0)-np.cross(com,[0,0,W])]
            worst=max(worst,float(np.max(abs(residual))))
            mean_point_counterexample=max(mean_point_counterexample,float(np.linalg.norm(np.cross(P,F).sum(0)-np.cross(P.mean(0),F.sum(0)))))
            cases+=1
    assert worst<1e-12 and mean_point_counterexample>.01
    return dict(passed=True,asymmetric_four_contact_cases=cases,maximum_wrench_residual=worst,
                incorrect_single_mean_contact_moment_error_Nm=mean_point_counterexample,
                scope='Force allocation algebra only; actual full23joint derivatives and virtual work independently checked at original standing before the root',robot_states=0,physics_steps=0)

if __name__=='__main__':
    result=audit()
    with Path(__file__).with_name('synthetic_audit.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
