"""NEW two-stage whole-path interior-witness execution, not478 continuation."""
from low_seat_free_gap_execution_472 import *
from low_seat_joint_ray_476 import decode
from low_seat_interior_witness_482 import assemble_witness,certify,save as save_interior
from fractions import Fraction as F
SPEC=dict(SPEC)
SPEC.update(max_linear_programs=144,max_maps=120,seed='EXACT sealed472 INITIAL, inherited without a repeated lift; no478 state or rejected witness.',budget_justification='24 predictors allow18 half-reductions plus6 rejection opportunities. Each has1 primary+1 interior LP and at most4 interior corrections:144LPs,120maps. This is a new two-stage formulation, not an old cap extension.',witness_objective='Maximize relative finite-domain clearance subject to all hard rows and half the modeled primary improvement; exact positive unscaled certificate required.',prediction='Primary vector never decoded. Acceptance prediction uses the final certified interior direction. Corrections retain half that improvement.',finite_map='Unchanged476 decoder must have EXACT alpha1 and strictly interior decoded p/load/drop values. No witness editing.',physical_admission=False)

def solve_stage(a,folder,role,ceiling=None):
    folder.mkdir(exist_ok=False);save_assembly(a,folder/'base')
    (folder/'stage.json').write_text(json.dumps(dict(role=role,soft_ceiling=ceiling)))
    started=time.monotonic();w=None
    try:
        if np.any(a['full_step_lower']>a['full_step_upper']):raise ValueError('empty full-control correction interval')
        if role=='primary':p=a
        else:
            w=assemble_witness(a['_d'],a,ceiling);save_interior(w,folder/'interior');p=w
    except ValueError as exc:
        meta=dict(status=-10,candidate=False,success=False,actual_lp_call=False,message=str(exc),elapsed_s=time.monotonic()-started)
        (folder/'solver.json').write_text(json.dumps(meta,indent=2));return None,meta,w
    res=linprog(p['cost'],A_ub=p['matrix'],b_ub=p['rhs'],A_eq=p['equality'],b_eq=p['equality_rhs'],bounds=p['bounds'],method=SPEC['lp_method'],options=dict(time_limit=SPEC['lp_time_limit_s'],primal_feasibility_tolerance=SPEC['lp_primal_dual_tolerance'],dual_feasibility_tolerance=SPEC['lp_primal_dual_tolerance']))
    meta=dict(status=int(res.status),message=str(res.message),candidate=res.x is not None,success=bool(res.success),iterations=int(res.nit),actual_lp_call=True,role=role,elapsed_s=time.monotonic()-started)
    if res.x is not None:
        v=res.x;violation=max(0.,float((p['matrix']@v-p['rhs']).max()),float(abs(p['equality']@v-p['equality_rhs']).max()),float((p['bounds'][:,0]-v).max()),float((v-p['bounds'][:,1]).max()))
        meta.update(independent_linear_violation=violation,linear_pass=bool(np.isfinite(v).all() and violation<=SPEC['independent_linear_bound']),objective=float(p['cost']@v))
        np.save(folder/'witness.npy',v)
    (folder/'solver.json').write_text(json.dumps(meta,indent=2));return res,meta,w

class InteriorRetraction(GapRetraction):
    def trial(self,a,tangent,folder):
        self.counts['finite_maps']+=1;assert self.counts['finite_maps']<=SPEC['max_maps']
        target,drops,record=decode(self.d,a,tangent)
        np.savez_compressed(folder/'mapped.npz',target=target,tangent=tangent)
        lo,hi=map_interval(self.d,a);strict=True
        for i,v in enumerate(self.d['free']):
            b,s=divmod(int(v),10)
            if s and not lo[i]<target[b,s]<hi[i]:strict=False
        drop_strict=all(l<x<u for l,x,u in zip(self.d['gap_domain']['lower'],drops,self.d['gap_domain']['upper']))
        record.update(alpha_exact_one=unpack(record['alpha'])==F(1),decoded_targets_strictly_interior=bool(strict),decoded_drops_strictly_interior=bool(drop_strict))
        (folder/'mapped.json').write_text(json.dumps(record))
        if not(record['alpha_exact_one'] and strict and drop_strict):raise ValueError('Certified-ray finite encoding contract failed before root')
        x=a['X']+np.einsum('nic,nc->ni',a['T'],target-a['target'])
        out=self.lift(x,target,folder);out.update(target=target,drops=drops,map_record=record);return out
