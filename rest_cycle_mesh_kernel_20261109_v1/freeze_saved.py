"""Freeze one saved-geometry evaluation after independent synthetic review."""
import hashlib,json,sys,sysconfig
from pathlib import Path


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(project,work):
    project=Path(project);work=Path(work);old=project/'.cpu-work/20261109/contact_path'
    prior=json.loads((old/'partitioned_freeze.json').read_text())
    inputs={k:prior['inputs'][k] for k in ['frame','model','colliders','furniture_mesh','target_region']}
    inputs.update(plan=str(old/'controller_review/bound_plan.json'),review_log=str(work/'strict_review/tests.log'),
        benchmark=str(work/'numpy_review/synthetic_benchmark.json'),compiler_record=str(work/'numpy_review/software_accounting.json'))
    review=Path(inputs['review_log']).read_text();benchmark=json.loads(Path(inputs['benchmark']).read_text())
    if not review.rstrip().endswith('OK') or 'Ran 13 tests' not in review:raise ValueError('INDEPENDENT_REVIEW_REQUIRED')
    if not benchmark['exact_classifications_equal'] or benchmark['speedup']<2:raise ValueError('MEASURED_SAVINGS_MECHANISM_REQUIRED')
    library=work/('numpy_clip'+sysconfig.get_config_var('EXT_SUFFIX'))
    sources=[work/name for name in ['numpy_clip.cpp','mesh_audit.py','parallel_audit.py','test_mesh_kernel.py','freeze_saved.py','run_saved.py']]
    sources.extend([library,project/'sitting_task/src/humanoid_sitting/shared_frame.py',project/'experiments/furniture_repair_20261106_v1/audit_mesh_clearance.py'])
    account=[old/p/'process_accounting.json' for p in ['execution','partitioned_execution','mesh_completion']]
    import numpy,scipy
    contract=dict(decision='ONE_COMPILED_SAVED_MESH_AUDIT_NO_MOTION',project=str(project),inputs=inputs,library=str(library),
        sha256={str(p):sha(p) for p in [*inputs.values(),*sources,*account]},
        runtime=dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__),
        authorization='Human go continues CPU work on collision bottleneck; both GPUs reserved, no PPO or simulation.',
        scope='New compiled-loop execution of all saved-path collision predicates; no reconstruction or replay of completed foot/command proofs. Previous three timeouts remain closed.',
        association='Bound 640-control outward/return path, its captured frame, original collider meshes, exact furniture mesh and allowed triangle IDs.',
        mechanism='NumPy matmul/add/sub and scalar types retained; C++ replaces Python all/edge loops with same double operation order, no fast-math or FMA. Each segment partition retains its four samples and only final partition includes the terminal sample.',
        proof_review=['Same NumPy input values, shapes, contiguous layout and plane order preserve distance calculations.',
            'Inside/outside comparisons and polygon ordering are identical; edge scalar subtraction/division/product/addition each round to double with contraction disabled.',
            '13 independent tests compare complete polygons bit-for-bit with original NumPy, boundary/degenerate cases, full reports, exact sample partition, failed workers and bounded cleanup.',
            'Synthetic benchmark establishes speedup only; no claim of actual-path speed until execution.'],
        measured_synthetic_speedup=benchmark['speedup'],
        test='All5121poses and all original candidate triangles; return complete original collision report or explicit failure. No missing samples accepted.',
        resources=dict(total_wall_cap_s=45,work_stop_s=42,CPUs=9,workers=8,parent_AS_GiB=8,worker_AS_GiB=2,aggregate_AS_GiB=24,
            nodes=1281,samples=5121,shapes=38,furniture_triangles=12,maximum_candidate_tests=5121*38*12,max_planes_per_shape=12000,
            output_MiB=16,GPU_contexts=0,simulation_steps=0,PPO_updates=0),
        historical_wall_s=sum(json.loads(p.read_text())['wall_through_evidence_s'] for p in account),
        runtime_admission=False,full_model_admission=False,physical_motion_admission=False,physical_success=False,
        finite_return='A complete collision report is not full model or runtime admission. Native load/ankle/current-camera gates and physical cycles remain. No retry, automatic continuation or runtime reset.',retry_authorized=False)
    path=work/'freeze.json'
    with path.open('x') as f:json.dump(contract,f,indent=2);f.write('\n')
    print(json.dumps(dict(freeze=str(path),hashes=len(contract['sha256']),decision=contract['decision'])))


if __name__=='__main__':main(*sys.argv[1:])
