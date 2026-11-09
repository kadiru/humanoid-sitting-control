"""Replay saved LP primal/dual witnesses and independent analytic joint work."""
from pathlib import Path
import datetime, hashlib, json
import numpy as np
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel, apply, transform

D = Path('/data/humanoid-sitting-perception/low_seat_contact_01')
O = D/'contact_force_authority_audit_105'
def read(p): return json.loads(Path(p).read_text())
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    O.mkdir(exist_ok=False)
    source = D/'contact_force_authority_104/results.json'
    data = read(source)
    for p, h in data['source_sha256'].items(): assert sha(p) == h, p
    witness = D/'contact_force_authority_104/lp_witnesses.npz'
    a = np.load(witness)
    path = read(D/'stance_com_path_v2_98/results.json')['dense']
    m = BodyModel(D/'cooked_model_50/model.json')
    inertia = read(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json')
    W = sum(inertia['mass'])*9.81
    names = m.names
    moving = [names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']]
    moving += [names.index(s+'_'+j+'_joint') for s in ['left','right'] for j in ['shoulder_pitch','elbow_pitch']]
    kp = np.array([200,150,150,200,160,160]*2+[40]*4)
    by = {j['name']:j for j in m.joints}
    children = {}
    for j in m.joints:
        if j['body0']: children.setdefault(j['body0'][0], []).append(j['body1'][0])
    def descend(b):
        out = {b}
        for c in children.get(b, []): out |= descend(c)
        return out
    sub = {n:descend(by[n]['body1'][0]) for n in names}
    rows = []
    for k, (r, e, v) in enumerate(zip(data['nodes'], path['path'], path['command_nodes'])):
        U,rhs,A,b,x,c,bounds = [a[f'{k}_{n}'] for n in ['U','rhs','A','b','x','objective','bounds']]
        zi,ze,zl,zu = [np.array(r[n]) for n in ['solver_inequality_duals','solver_equality_duals','solver_lower_duals','solver_upper_duals']]
        lo,hi = bounds.T
        primal = max(max(abs(A@x-b)), max(U@x-rhs), max(lo-x), max(x-hi), 0.)
        signs = max(max(zi), max(-zl), max(zu), 0.)
        stationarity = max(abs(c-U.T@zi-A.T@ze-zl-zu))
        dual = float(rhs@zi+b@ze+lo@zl+hi@zu)
        gap = abs(float(c@x)-dual)
        assert primal < 1e-8 and signs < 1e-10 and stationarity < 1e-8 and gap < 1e-8
        q, root = np.array(e['q']), np.array(e['root'])
        T = transform(Rotation.from_quat(root[3:]).as_matrix(), root[:3])
        fk = m.fk(q); P = np.array(v['corner_world_points']); sides = v['corner_sides']
        B = []
        for j in moving:
            joint = by[names[j]]
            J = T@fk[joint['body0'][0]]@m.frames[names[j]][0]
            axis = J[:3,'XYZ'.index(joint['axis'])]
            B.append([-axis@np.cross(p-J[:3,3],[0,0,W]) if s+'_ankle_roll_link' in sub[names[j]] else 0. for p,s in zip(P,sides)])
        B = np.array(B)
        work = float(max(abs(B-U[32:48,:-1]).ravel()))
        assert work < 1e-6
        assert max(abs(B/kp[:,None]-U[:16,:-1]).ravel()) < 2e-8
        assert abs(-float(c@x)-r['best_margin_rad']) < 1e-8
        rows.append(dict(knot=k, primal_residual=float(primal), dual_stationarity=float(stationarity),
                         duality_gap=gap, analytic_virtual_work_error_Nm=work,
                         actual_command_limit_satisfied=r['actual_command_limit_satisfied']))
    assert len(rows)==115
    failures=[r['knot'] for r in rows if not r['actual_command_limit_satisfied']]
    assert failures==data['summary']['failing_actual_limit_knots']
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,
             scope='Independent analytic cross-product joint-work columns plus all115 saved LP primal/dual stationarity, sign, residual and objective checks. No LP rerun or physics. Numerical witnesses establish infeasibility of the fixed poses and fixed loads at knots99/100 within this model; not impossibility of other motions or dynamic safety.',
             nodes=rows, failing_actual_limit_knots=failures,
             max_primal_residual=max(r['primal_residual'] for r in rows),
             max_stationarity=max(r['dual_stationarity'] for r in rows),
             max_duality_gap=max(r['duality_gap'] for r in rows),
             max_analytic_work_error_Nm=max(r['analytic_virtual_work_error_Nm'] for r in rows),
             source_sha256={str(p.resolve()):sha(p) for p in [source,witness,Path(__file__)]},
             physics_ticks=0,optimizer_updates=0,holdout_used=False,physical_admission=False)
    (O/'independent_audit.json').write_text(json.dumps(out,indent=2))
    (O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
    print(json.dumps({k:v for k,v in out.items() if k not in ['nodes','source_sha256']}),flush=True)

if __name__=='__main__': main()
