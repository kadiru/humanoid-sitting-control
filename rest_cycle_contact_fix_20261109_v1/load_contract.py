"""The existing spatial-wrench uncertainty, carried unchanged to the controller.

This is a producer/consumer compatibility check, not sensor calibration. A
square wrench solve does not establish the physical validity of its interval.
No interval is narrowed, clipped, or replaced by a geometric zero here.
"""
import math
from fractions import Fraction

LOAD_RADIUS_BW = .02
RELEASE_THRESHOLD_BW = .01


def estimate_interval(estimate_bw, *, available):
    if not available or estimate_bw is None:
        return None, None
    estimate=float(estimate_bw)
    if not math.isfinite(estimate):
        return None, None
    lower,upper=estimate-LOAD_RADIUS_BW,estimate+LOAD_RADIUS_BW
    if not math.isfinite(lower) or not math.isfinite(upper) or upper<0:
        return None,None
    return lower,upper


def release_compatibility():
    # Exact rational interpretation of the existing binary64 constants.
    radius=Fraction.from_float(LOAD_RADIUS_BW)
    threshold=Fraction.from_float(RELEASE_THRESHOLD_BW)
    # Consumer requires 0 <= estimate-radius and estimate+radius < threshold.
    # Hence estimate >= radius AND estimate < threshold-radius.
    possible=radius < threshold-radius
    zero=estimate_interval(0.,available=True)
    return dict(passed=possible,producer='existing spatial wrench estimate +/-0.02BW',
        lower_estimate_bound_inclusive_bw=float(radius),
        upper_estimate_bound_exclusive_bw=float(threshold-radius),
        required_interval_width_strictly_below_bw=RELEASE_THRESHOLD_BW,
        producer_interval_width_bw=2*LOAD_RADIUS_BW,
        zero_estimate_interval_bw=list(zero),
        reason='COMPATIBLE' if possible else 'NO_ESTIMATE_CAN_SATISFY_RELEASE_INTERVAL',
        loaded_admission_is_not_release_admission=True,
        interval_narrowed=False,physical_success=False)


def validation_verdict(*, actuator_filter_passed, idealized_protocol_passed):
    compatibility=release_compatibility()
    return dict(passed=bool(actuator_filter_passed and idealized_protocol_passed and compatibility['passed']),
        actuator_filter_passed=bool(actuator_filter_passed),
        idealized_sensor_protocol_passed=bool(idealized_protocol_passed),
        producer_to_controller_compatible=compatibility['passed'],
        release_contract=compatibility,physical_success=False,path_admission_changed=False)
