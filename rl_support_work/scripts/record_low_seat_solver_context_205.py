"""Read-only source/documentation context; no runtime import or simulation."""
from pathlib import Path
import datetime,hashlib,json
R=Path('/home/kadir/humanoid-sitting');D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'solver_context_205'
def source(p,a,b):
    p=R/p;s=p.read_text().splitlines()
    return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),line_start=a,line_end=b,text='\n'.join(s[a-1:b]))
def main():
    O.mkdir(exist_ok=False)
    records=[source('sitting_task/src/humanoid_sitting/tasks/config.py',1,27),
        source('IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py',294,315),
        source('IsaacLab/source/isaaclab_physx/isaaclab_physx/physics/physx_manager_cfg.py',60,72),
        source('IsaacLab/source/isaaclab_physx/isaaclab_physx/physics/physx_manager_cfg.py',162,178),
        source('IsaacLab/source/isaaclab_physx/isaaclab_physx/physics/physx_manager.py',898,912)]
    url='https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/Simulation.html'
    summary=('NVIDIA documents that constraint-solver velocity outputs need not equal pose differences divided by timestep: position and velocity iterations address different errors. It separately describes TGS steady-state velocity bias when external forces are applied once per frame rather than each solver iteration. These mechanisms are useful hypotheses, not proof of the cause in this robot. The installed source defaults already enable external forces each iteration, so the documented disabled-flag example must not be assumed applicable. No solver parameter or settling criterion was changed.')
    result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,primary_documentation=dict(url=url,sections=['Solver Iterations','TGS Steady-State Velocity and Position Discrepancy'],summary=summary),installed_source_evidence=records,
        source_defaults=dict(task_uses_G1_CFG=True,G1_position_iterations=8,G1_velocity_iterations=4,PhysxCfg_solver_type=1,PhysxCfg_enable_external_forces_every_iteration=True),
        live_solver_parameters_captured=False,causal_mechanism_proven=False,physics_ticks=0,optimizer_updates=0,physical_admission=False,
        decision='No sensor substitution, raw-velocity guard waiver, criterion relaxation or physics-setting change follows this context. Resolve runtime semantics and preserve exact matched comparisons before a separately scoped controller revision.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (O/'context.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({'context.json':hashlib.sha256((O/'context.json').read_bytes()).hexdigest()},indent=2));print(json.dumps(dict(physics_ticks=0,live_parameters_verified=False,source_defaults=result['source_defaults'])))
if __name__=='__main__':main()
