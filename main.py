# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.




import os
import numpy as np
import matchms.filtering as ms_filters
from matchms import calculate_scores
from matchms.similarity import CosineGreedy
from matchms.importing import load_from_mgf
from matchms.similarity import ModifiedCosine
from matplotlib import pyplot as plt
import time


def metadata_processing(spectrum):
    spectrum = ms_filters.default_filters(spectrum)
    spectrum = ms_filters.repair_inchi_inchikey_smiles(spectrum)
    spectrum = ms_filters.derive_inchi_from_smiles(spectrum)
    spectrum = ms_filters.derive_smiles_from_inchi(spectrum)
    spectrum = ms_filters.derive_inchikey_from_inchi(spectrum)
    spectrum = ms_filters.harmonize_undefined_smiles(spectrum)
    spectrum = ms_filters.harmonize_undefined_inchi(spectrum)
    spectrum = ms_filters.harmonize_undefined_inchikey(spectrum)
    spectrum = ms_filters.add_precursor_mz(spectrum)
    return spectrum
def peak_processing(spectrum):
    spectrum = ms_filters.default_filters(spectrum)
    spectrum = ms_filters.normalize_intensities(spectrum)
    spectrum = ms_filters.select_by_intensity(spectrum, intensity_from=0.01)
    spectrum = ms_filters.select_by_mz(spectrum, mz_from=10, mz_to=1000)
    return spectrum
similarity_measure = ModifiedCosine(tolerance=0.005)
path_data = "C:/Users/usuario/Desktop/GSOC/vector_db"  # enter path to downloaded mgf file
file_mgf = os.path.join(path_data, 
                        "GNPS-NIH-NATURALPRODUCTSLIBRARY.mgf")
spectrums = list(load_from_mgf(file_mgf))
spectrums = [metadata_processing(s) for s in spectrums]
spectrums = [peak_processing(s) for s in spectrums]
start_time = time.time()
scores = calculate_scores(spectrums, spectrums, similarity_measure, is_symmetric=True)
scores_array = scores.scores.to_array()
min_match = 5
plt.figure(figsize=(6,6), dpi=150)
plt.imshow(scores_array[:50, :50]["ModifiedCosine_score"]
           * (scores_array[:50, :50]["ModifiedCosine_matches"] >= min_match), cmap="viridis")
plt.colorbar(shrink=0.7)
plt.title("Modified Cosine spectra similarities (min_match=5)")
plt.xlabel("Spectrum #ID")
plt.ylabel("Spectrum #ID")
plt.show()
modified_cosine = time.time()
modified_cosine_time = modified_cosine - start_time
similarity_measure = CosineGreedy(tolerance=0.005)
scores = calculate_scores(spectrums, spectrums, similarity_measure, is_symmetric=True)
scores_array = scores.scores.to_array()
best_matches = scores.scores_by_query(spectrums[5], name="CosineGreedy_score", sort=True)[:10]
print([x[1] for x in best_matches])
cosine_greedy = time.time()
cosine_greedy_time = cosine_greedy-modified_cosine
print("Modified cosine time:", modified_cosine_time, "seconds")
print("Cosine greedy time:", cosine_greedy_time, "seconds")
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
