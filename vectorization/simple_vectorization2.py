import numpy as np


def simpleVectorization2(spectra):
    spectra_array = np.empty((len(spectra), 10000))
    i=0
    for s in spectra:
        vector = np.empty((2, len(s.peaks.mz)))
        vector[0,] = s.peaks.mz
        vector[1,] = s.peaks.intensities
        for j in range(vector.shape[1]):
            mz = int(vector[0][j]*10)
            spectra_array[i][mz] = vector[1][j]
        i= i+1
    return vector
