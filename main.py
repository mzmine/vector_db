# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.





from IO.export import generateNetwork
from IO.importing import loadScpectrums
from IO.importing import importSpec2VecModel
from IO.importing import importMS2DeepscoreModel
from preprocessing.metadata_processing import metadata_processing
from preprocessing.peak_processing import peak_processing
from preprocessing.train_spec2vec import train_spec2vec
from vectorization.simple_vectorization import simpleVectorization
from comparison.cosine_greedy import cosine_greedy
from comparison.modified_cosine import modified_cosine
from comparison.spec2vec import spec2vec
from comparison.ms2deepscore import ms2deepscore
from visualization.plot_scores_restrictive import plotScoresRestrictive
from visualization.plot_scores import plotScores
from visualization.get_best_matches import getBestMatches





spectrums= loadScpectrums("C:\\Users\\usuario\\Desktop\\GSOC\\vector_db\\GNPS-NIH-NATURALPRODUCTSLIBRARY.mgf")
spectrums = [metadata_processing(s) for s in spectrums]
spectrums = [peak_processing(s) for s in spectrums]
vectors = [simpleVectorization(s) for s in spectrums]
print(vectors)
model = importMS2DeepscoreModel("C:\\Users\\usuario\\Desktop\\GSOC\\vector_db\\MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5")
scores = ms2deepscore(model, spectrums)
getBestMatches(scores, spectrums, 11, 10)
generateNetwork(scores, spectrums)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
