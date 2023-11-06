import numpy as np


def simple_vectorization(spectra):
    vectors = []
    for spectrum in spectra:
        vector = np.empty((2, len(spectrum.peaks.mz)))
        vector[0, ] = spectrum.peaks.mz
        vector[1, ] = spectrum.peaks.intensities
        vectors.append(vector)
    max_vector_length = max(v.shape[1] for v in vectors)
    padded_vectors = []
    for v in vectors:
        if v.shape[1] < max_vector_length:
            padding = max_vector_length - v.shape[1]
            padded_vector = np.pad(v, [(0, 0), (0, padding)], mode='constant')
            padded_vectors.append(padded_vector)
        else:
            padded_vectors.append(v)
    vectors_array = np.array(padded_vectors).reshape(-1, 2 * max_vector_length)
    return vectors_array

