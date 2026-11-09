"""Request a current camera exposure without advancing physics.

In the installed SensorBase, force_recompute flushes the due mask but does not
make a sensor due. Eager 100-ms camera updates can therefore precede a 100-ms
controller read by 95 ms. Temporarily setting the update period to zero makes
this explicit acquisition due; the original schedule is restored even on error.
"""
def acquire_current_camera(camera):
    period=camera.cfg.update_period
    try:
        camera.cfg.update_period=0.
        camera.update(0.,force_recompute=True)
    finally:
        camera.cfg.update_period=period
