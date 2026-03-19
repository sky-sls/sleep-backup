"""
Set of functions for down- and re-sampling of PSG signals
"""
import numpy as np
from mne.filter import resample, notch_filter, filter_data
from scipy.signal import resample_poly

def fourier_resample(psg, new_sample_rate, old_sample_rate):
    psg = resample(psg, new_sample_rate, old_sample_rate, axis=0)
    # psg = notch_filter(psg, new_sample_rate, np.arange(60, 241, 60))  # ljy改：去除工频干扰
    # psg = filter_data(psg, new_sample_rate, 0.1, 50.)  # 带通滤波去噪
    return psg


def poly_resample(psg, new_sample_rate, old_sample_rate):
    psg = resample_poly(psg, new_sample_rate, old_sample_rate, axis=0)
    # psg = notch_filter(psg, new_sample_rate, np.arange(60, 241, 60))  # ljy改：去除工频干扰
    # psg = filter_data(psg, new_sample_rate, 0.1, 50.)  # 带通滤波去噪
    return psg


def set_psg_sample_rate(psg, new_sample_rate, old_sample_rate, method="poly"):
    """
    Resamples PSG of sample rate 'old_sample_rate' to new sample rate
    'new_sample_rate'. The length of the PSG will be a factor
    new_sample_rate/old_sample_rate the original.

    Args:
        psg:              ndarray of PSG data to resample, shape [N, C].
                          Resampling is performed over axis 0 (sample dim)
        new_sample_rate:  Sample rate of the new signal
        old_sample_rate:  Sample rate of the original signal
        method:           Resampling method, one of poly, fourier

    Returns:
        A resampled PSG ndarray
    """
    new_sample_rate = int(new_sample_rate)
    old_sample_rate = int(old_sample_rate)
    method = method.lower().split("_")[0]
    if method == "poly":
        return poly_resample(psg, new_sample_rate, old_sample_rate)
    elif method == "fourier":
        return fourier_resample(psg, new_sample_rate, old_sample_rate)
    else:
        raise ValueError("Invalid method {} selected.".format(method))
