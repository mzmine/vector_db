import numpy as np


def simple_vectorization(spectrum):
    vector = np.empty((2, len(spectrum.peaks.mz)))
    vector[0, ] = spectrum.peaks.mz
    vector[1, ] = spectrum.peaks.intensities
    return vector

