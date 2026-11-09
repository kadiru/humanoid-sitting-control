"""Install one read-only velocity trace hook in an isolated, pinned SDK clone.

Never point this at the installed Isaac Sim runtime. Compilation alone performs
no simulation; the diagnostic executable requires a separate bounded launch.
"""
import argparse
import hashlib
from pathlib import Path
import subprocess

REVISION='4f5f47733d72bc0966af6b1e469eb76acbae7d69'
RELATIVE='physx/source/lowleveldynamics/src/DyFeatherstoneArticulation.cpp'
FUNCTION='\tvoid FeatherstoneArticulation::recordDeltaMotionTGS('
NEEDLE='\t\tPxReal* PX_RESTRICT jointNewVelocities = data.getJointNewVelocities();'


def instrument(root):
    root=Path(root).resolve()
    actual=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    if actual!=REVISION:
        raise ValueError('PINNED_PHYSX_REVISION_REQUIRED')
    path=root/RELATIVE
    original=subprocess.check_output(['git','-C',str(root),'show','HEAD:'+RELATIVE])
    if path.read_bytes()!=original:
        raise ValueError('SDK_SOURCE_MUST_BE_UNMODIFIED')
    source=original.decode();start=source.index(FUNCTION)
    point=source.index(NEEDLE,start)+len(NEEDLE)
    end=source.index('\n\tvoid ',start+len(FUNCTION))
    if point>=end:
        raise ValueError('HOOK_OUTSIDE_EXPECTED_FUNCTION')
    declaration='extern "C" void rest_diagnostic_position_velocity(unsigned, const float*);\n'
    hook='\n\t\t::rest_diagnostic_position_velocity(data.getDofs(), jointNewVelocities);'
    changed=(declaration+source[:point]+hook+source[point:]).encode()
    path.write_bytes(changed)
    print('read_only_trace_hook',hashlib.sha256(original).hexdigest(),hashlib.sha256(changed).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('isolated_sdk_clone')
    instrument(parser.parse_args().isolated_sdk_clone)
