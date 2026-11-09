"""Read-only source/runtime metadata; no scientific arrays or model calls."""
import hashlib
import importlib.metadata as md
import json
import pathlib
import sys
import time

start = time.monotonic()
import numpy
import scipy.sparse
import torch
import scipy.spatial.transform
import humanoid_sitting.shared_frame
from scipy.ndimage import label
from scipy.optimize import least_squares

paths = {pathlib.Path(sys.executable).resolve()}
for module in list(sys.modules.values()):
    path = getattr(module, '__file__', None)
    if path:
        path = pathlib.Path(path).resolve()
        if path.is_file():
            paths.add(path)
for line in pathlib.Path('/proc/self/maps').read_text().splitlines():
    fields = line.split()
    if len(fields) >= 6 and fields[-1].startswith('/'):
        p = pathlib.Path(fields[-1]).resolve()
        if p.is_file():
            paths.add(p)
for name in ['numpy', 'scipy', 'torch', 'sympy', 'networkx', 'typing-extensions',
             'filelock', 'fsspec', 'jinja2', 'setuptools', 'mpmath', 'markupsafe']:
    package = md.distribution(name)
    paths.update(pathlib.Path(package.locate_file(x)).resolve()
                 for x in package.files if str(x).endswith(('.py', '.so', '.pth', '/METADATA', '/RECORD'))
                 and pathlib.Path(package.locate_file(x)).is_file())
def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as source:
        for b in iter(lambda:source.read(4194304), b''): h.update(b)
    return h.hexdigest()
R = pathlib.Path('/home/kadir/humanoid-sitting')
D = pathlib.Path('/data/humanoid-sitting-perception/low_seat_contact_01')
extra = [D/x for x in ['cooked_model_50/model.json','lattice_path_430/initial.npz',
    'clock_lift_v2_434/lift.npz','contact_lift_preparation_438/preparation.json']]
extra += [D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json',
    R/'sitting_task/src/humanoid_sitting/shared_frame.py',
    R/'sitting_task/src/humanoid_sitting/__init__.py']
print(json.dumps(dict(scope=__doc__, scientific_calls=0,
    versions={k:md.version(k) for k in ['numpy','scipy','torch']},
    python=sys.version, runtime_sha256={str(p):sha(p) for p in sorted(paths)},
    runtime_bytes=sum(p.stat().st_size for p in paths),
    input_sha256={str(p):sha(p) for p in extra},
    elapsed_s=time.monotonic()-start), indent=2))
