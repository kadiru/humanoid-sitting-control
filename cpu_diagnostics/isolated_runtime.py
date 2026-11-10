"""Headless Kit/PhysX bootstrap; requires OS-level GPU device isolation."""
import glob
import os
from pathlib import Path


def require_gpu_isolation():
    nodes = glob.glob('/dev/nvidia*') + glob.glob('/dev/dri/*')
    if (not Path('/.dockerenv').exists() or nodes
            or os.environ.get('CUDA_VISIBLE_DEVICES') != ''
            or os.environ.get('NVIDIA_VISIBLE_DEVICES') != 'void'):
        raise RuntimeError('OS_LEVEL_GPU_ISOLATION_REQUIRED')


def start():
    require_gpu_isolation()
    import isaacsim
    import carb
    import omni.kit.app
    root = Path(isaacsim.__file__).parent
    carb.get_framework().load_plugins(
        loaded_file_wildcards=['omni.kit.app.plugin'],
        search_paths=[str(root/'kit/kernel/plugins')])
    app = omni.kit.app.get_app()
    app.startup('resting-cpu', str(root/'kit'), [
        str(Path(__file__)), str(Path(__file__).with_name('physics_cpu.kit')),
        '--ext-folder', str(root/'extscache'), '--ext-folder', str(root/'exts'),
        '--portable-root', '/tmp/resting-kit', '--no-window',
        '--/plugins/carb.tasking.plugin/threadCount=4',
        '--/plugins/omni.tbb.globalcontrol/maxThreadCount=4',
        '--/app/renderer/enabled=false', '--/app/fastShutdown=false',
        '--/log/file=/tmp/resting_cpu_kit.log'])
    return app
