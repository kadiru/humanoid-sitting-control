"""Build the NumPy-compatible CPU extension without fast-math or contraction.

Usage: python build.py OUTPUT_DIRECTORY
The output directory must be new. Building does not run a geometry audit.
"""
import hashlib,json,os,subprocess,sys,sysconfig,time
from pathlib import Path


def main(output):
    os.environ.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    import numpy
    output=Path(output);output.mkdir(exist_ok=False)
    source=Path(__file__).resolve().with_name('numpy_clip.cpp')
    library=output/('numpy_clip'+sysconfig.get_config_var('EXT_SUFFIX'))
    command=['c++','-std=c++17','-O3','-fPIC','-shared','-fno-fast-math','-ffp-contract=off',
             '-I'+numpy.get_include(),'-I'+sysconfig.get_path('include'),str(source),'-o',str(library)]
    start=time.monotonic();run=subprocess.run(command,capture_output=True,text=True,timeout=45)
    (output/'compile.log').write_text(run.stdout+run.stderr)
    record=dict(command=command,wall_s=time.monotonic()-start,exit_code=run.returncode,
        python=sys.version,numpy=numpy.__version__,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        library_sha256=hashlib.sha256(library.read_bytes()).hexdigest() if run.returncode==0 else None,
        GPU_contexts=0,geometry_audits=0,simulation_steps=0,PPO_updates=0,
        independent_regression_review_required=True)
    (output/'build.json').write_text(json.dumps(record,indent=2)+'\n')
    if run.returncode:raise RuntimeError('CPU_EXTENSION_BUILD_FAILED; see compile.log')
    print(library)


if __name__=='__main__':main(*sys.argv[1:])
