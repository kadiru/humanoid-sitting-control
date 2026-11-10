"""Verify the installed PhysX Python runtime in a GPU-inaccessible container.

No stage, robot, physics step or camera is created. This is an infrastructure
check, not a sensor correction or physical test. Run only in the declared
unprivileged runc container; environment masking by itself is insufficient.
"""
import json
import sys
import time
from pathlib import Path
from isolated_runtime import start, require_gpu_isolation

START = time.monotonic()


def main():
    out = Path(sys.argv[1])
    out.mkdir(exist_ok=False)
    require_gpu_isolation()
    result = dict(GPU_device_nodes=[], OS_GPU_isolation=True,
                  physics_steps=0, robot_trials=0, sensor_repair=False)
    app = None
    try:
        app = start()
        import omni.physx
        import omni.physics.tensors
        from pxr import PhysxSchema
        result.update(runtime_available=True,
            physx_module=str(Path(omni.physx.__file__).resolve()),
            simulation_interface_available=bool(omni.physx.get_physx_simulation_interface()),
            physics_schema_available=hasattr(PhysxSchema, 'PhysxSceneAPI'))
    except BaseException as exc:
        result.update(runtime_available=False, error=repr(exc))
        raise
    finally:
        result['wall_s'] = time.monotonic()-START
        (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)
        if app is not None:
            app.shutdown()


if __name__ == '__main__':
    main()
