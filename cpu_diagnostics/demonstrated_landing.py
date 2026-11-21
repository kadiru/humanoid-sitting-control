"""Recognize the support where the nominal arm approach was demonstrated cleanly."""
import numpy as np


def select_landing(rim_frames, reference):
    # The motion worker records rear-camera edge, height and pitch, with NaNs
    # on indirect observations. Compare only direct readings to the demonstration.
    features = np.asarray(rim_frames,dtype=float)
    direct = np.isfinite(features).all(axis=2)
    tolerances = np.asarray(reference['tolerances_m_m_deg'])
    error = np.abs(features - np.asarray(reference['rim_m_m_deg']))
    matched = direct & (error <= tolerances).all(axis=2)
    return matched.sum(axis=0) >= 4
