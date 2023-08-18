import numpy as np
from numba import njit


@njit
def _bin_peak_list(peak_list: np.array, min_mz: float = 11, max_mz: float = 1500, bin_step: float = 0.05, precursor_mz:
float = None, include_neutral_loss: bool = False) -> list:
    mzs, intensities = peak_list

    # start at min mz
    bin_lb = min_mz
    bin_ub = min_mz + bin_step
    binned_pl = []
    while bin_ub <= max_mz:
        bin_intensity = 0.
        for i, mz in enumerate(mzs):
            if bin_lb < mz <= bin_ub:
                bin_intensity += intensities[i]
            # neutral loss
            if include_neutral_loss and (bin_lb < precursor_mz-mz <= bin_ub):
                bin_intensity += intensities[i]
        binned_pl.append(bin_intensity)
        bin_ub += bin_step
        bin_lb += bin_step

    return binned_pl


def bin_peak_list(peak_list: np.array, min_mz: float = 11,  max_mz: float = 1500, bin_step: float = 0.05,
                  precursor_mz=None, include_neutral_loss: bool = False) -> np.array:
    return np.array(_bin_peak_list(peak_list, min_mz, max_mz, bin_step, precursor_mz, include_neutral_loss),
                    dtype="float32")