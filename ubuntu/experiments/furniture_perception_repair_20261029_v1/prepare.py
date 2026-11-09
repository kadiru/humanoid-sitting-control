"""CPU preparation: bind inputs, program and independently defined audits."""
import ast,time
from paths import ROOT,HERE,OUT,ASSETS,PRIOR,PPO,BODY,CHECK,read,write,sha

def main():
    assert not (OUT/'receipt.json').exists()
    for p in HERE.glob('*.py'):ast.parse(p.read_text(),filename=str(p))
    assert read(HERE/'cpu_contract_checks.json')['passed']
    # Explicit interface review: recognition accepts no asset ID/world pose/truth.
    tree=ast.parse((HERE/'perception.py').read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='observe')
    assert [a.arg for a in fn.args.args]==['model','K','T','depth','q','gravity','intent','prior']
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['read','open','eval','exec'] for n in ast.walk(fn))
    params=read(HERE/'protocol.json')['resources']
    assert params['maximum_physics_steps']==15*(1000+4*1025)+1000+3600
    assert params['maximum_control_instance_steps']==(72+14)*1025
    assert params['maximum_simulator_launches']==16
    # Start from the current experiment's frozen transitive dependencies, never
    # changing them. Add the separate leaning interfaces and furniture inputs.
    files=set(read(BODY/'freeze.json'))
    for folder in [HERE,PRIOR,CHECK,ASSETS,ROOT/'experiments/full_body_contact_trainer_4090_step_v5']:
        files.update(str(p) for p in folder.glob('*.py'))
    for c in read(ASSETS/'cases.json'):
        files.update([c['mesh_npz'],c['mesh_usd'],str(ASSETS/'scenes'/c['id']/'target_region.json')])
    for task in ['sitting','leaning']:
        p=ROOT/'outputs/full_body_contact_imitation_v4/dataset'/task
        files.update(str(p/n) for n in ['bank.npz','metadata.json','geometry.json'])
    files.update(str(HERE/n) for n in ['protocol.json','cpu_contract_checks.json','render_colliders.json'])
    files.add(str(ROOT/'outputs/full_body_contact_imitation_v5/train_leaning_64111/update_256.pt'))
    for p,h in read(BODY/'freeze.json').items():assert sha(p)==h,('PPO_INPUT_CHANGED',p)
    review=dict(passed=True,review_kind='Single execution owner source review with separately implemented NumPy/ray/force audit programs; no external reviewer claimed.',
        interfaces_no_actor_truth=True,capture_pose_audit_only=True,pixel_centers=True,connected_components=True,
        camera_origin_normal_orientation=True,original_plane_gate_retained=True,curved_surface_membership_reported_separately=True,
        no_physical_threshold_changes=True,no_action_changes=True,wait_for_both_seeds_and_all_evaluations=True,
        gpu4090_only=True,no_new_training=True,no_automatic_retries=True,no_recurring_automation=True,
        actual_resource_bounds_checked=True,partial_or_failed_outcomes_preserved=True,
        audits=['audit_depth.py','audit_support.py',str(PRIOR/'audit.py')],
        known_limitations=['Curved furniture is represented by local planes; old globalplane matching can fail despite correct face recognition.',
        'Sensorselected development motion library is not a novelfurniture generalization guarantee.',
        'Body fixtures are held under controlled loads and do not prove wholebodybalance.',
        'Contact channels touching multiple normals remain unvalidated, never clean by relaxation.'])
    write(HERE/'source_review.json',review);files.add(str(HERE/'source_review.json'))
    bindings={p:sha(p) for p in sorted(files)};write(HERE/'freeze.json',bindings)
    write(HERE/'preparation.json',dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),files=len(bindings),freeze_sha256=sha(HERE/'freeze.json'),scientific_launches=0))
    print('Frozen',len(bindings),'files. New simulator launches:0.')
if __name__=='__main__':main()
