# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.




import os
import numpy as np
from IO.export import generateNetwork
from IO.importing import loadScpectrums
from IO.importing import importSpec2VecModel
from IO.importing import importMS2DeepscoreModel
from preprocessing.metadata_processing import metadata_processing
from preprocessing.peak_processing import peak_processing
from preprocessing.train_spec2vec import train_spec2vec
from comparison.cosine_greedy import cosine_greedy
from comparison.modified_cosine import modified_cosine
from comparison.spec2vec import spec2vec
from comparison.ms2deepscore import ms2deepscore
from visualization.plot_scores_restrictive import plotScoresRestrictive
from visualization.plot_scores import plotScores
from visualization.get_best_matches import getBestMatches
import time
import gensim
from matchms.importing import load_from_json
import pyperf
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model



#TUTORIAL 1 modified cosine vs cosine greedy
"""
spectrums= loadScpectrums("C:\\Users\\usuario\\Desktop\\GSOC\\vector_db\\GNPS-NIH-NATURALPRODUCTSLIBRARY.mgf")
spectrums = [metadata_processing(s) for s in spectrums]
spectrums = [peak_processing(s) for s in spectrums]
start_time = time.time()
scores = modified_cosine(0.005,spectrums)
scores_array = scores.scores
min_match = 5
plotScoresRestrictive(scores_array, min_match)
modified_cosine = time.time()
modified_cosine_time = modified_cosine - start_time
scores=cosine_greedy(0.005,spectrums)
getBestMatches(scores, spectrums, 5, 10)
cosine_greedy = time.time()
cosine_greedy_time = cosine_greedy-modified_cosine
print("Modified cosine time:", modified_cosine_time, "seconds")
print("Cosine greedy time:", cosine_greedy_time, "seconds")
"""
#TUTORIAL 2 Spec2Vec
spectrums= loadScpectrums("C:\\Users\\usuario\\Desktop\\GSOC\\vector_db\\GNPS-NIH-NATURALPRODUCTSLIBRARY.mgf")
spectrums = [metadata_processing(s) for s in spectrums]
spectrums = [peak_processing(s) for s in spectrums]
model = importMS2DeepscoreModel("C:\\Users\\usuario\\Desktop\\GSOC\\vector_db\\MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5")
scores = ms2deepscore(model, spectrums)
getBestMatches(scores, spectrums, 11, 10)
generateNetwork(scores, spectrums)

"""
#TUTORIAL 3 Networking

path_data = "C:/Users/migue/OneDrive/Escritorio/GSOC/vector_db"
file_mgf = os.path.join(path_data, "GNPS-NIH-NATURALPRODUCTSLIBRARY.mgf")
spectrums = list(load_from_mgf(file_mgf))
spectrums = [metadata_processing(s) for s in spectrums]
spectrums = [peak_processing(s) for s in spectrums]
path_model = os.path.join(os.path.dirname(os.getcwd()), "vector_db")
filename_model = "spec2vec_AllPositive_ratio05_filtered_201101_iter_15.model"
filename = os.path.join(path_model, filename_model)
model = gensim.models.Word2Vec.load(filename)
spec2vec_similarity = Spec2Vec(model=model, intensity_weighting_power=0.5, allowed_missing_percentage=5.0)
scores = calculate_scores(spectrums, spectrums, spec2vec_similarity, is_symmetric=True)
generateNetwork(scores, spectrums)


#TUTORIAL 4 ms2deepscore
path_data = "C:/Users/migue/OneDrive/Escritorio/GSOC/vector_db"
file_mgf = os.path.join(path_data, "GNPS-NIH-NATURALPRODUCTSLIBRARY.mgf")
spectrums = list(load_from_mgf(file_mgf))
spectrums = [metadata_processing(s) for s in spectrums]
spectrums = [peak_processing(s) for s in spectrums]
model = load_model("MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5")
similarity_measure = MS2DeepScore(model)
scores = calculate_scores(spectrums, spectrums, similarity_measure, is_symmetric=True)
scores_array = scores.scores.to_array()
getBestMatches(scores, spectrums, 5, 10)
"""

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
