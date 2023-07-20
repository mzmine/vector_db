# This is a sample Python script.

# Press Mayús+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.




import numpy as np
from IO.export import generateNetwork
from IO.importing import loadScpectrums
from IO.importing import importSpec2VecModel
from IO.importing import importMS2DeepscoreModel
from preprocessing.metadata_processing import metadata_processing
from preprocessing.peak_processing import peak_processing
from preprocessing.train_spec2vec import train_spec2vec
from vectorization.reshape_vectors import reshape_vectors
from vectorization.create_IndexFlatL2 import create_IndexFlatL2
from vectorization.create_IndexFlatIP import create_IndexFlatIP
from vectorization.create_IndexIVFFlat import create_IndexIVFFlat
from vectorization.create_IndexLSH import create_IndexLSH
from vectorization.create_IndexHNSWFlat import create_IndexHNSWFlat
from vectorization.create_IndexIVFScalarQuantizer import create_IndexIVFScalarQuantizer
from vectorization.create_IndexIVFPQ import create_IndexIVFPQ
from vectorization.create_IndexIVFPQR import create_IndexIVFPQR
from vectorization.create_IndexPQ import create_IndexPQ
from vectorization.create_IndexScalarQuantizer import create_IndexScalarQuantizer
from vectorization.simple_vectorization import simpleVectorization
from comparison.cosine_greedy import cosine_greedy
from comparison.modified_cosine import modified_cosine
from comparison.spec2vec import spec2vec
from comparison.ms2deepscore import ms2deepscore
from visualization.plot_scores_restrictive import plotScoresRestrictive
from visualization.plot_scores import plotScores
from visualization.get_best_matches import getBestMatches
from statistics.export_benchmarking import export_benchmarking
import time



start_time = time.time()
spectrums= loadScpectrums("C:\\Users\\usuario\\Desktop\\GSOC\\vector_db\\GNPS-NIH-NATURALPRODUCTSLIBRARY.mgf")
spectrums = [metadata_processing(s) for s in spectrums]
spectrums = [peak_processing(s) for s in spectrums]
preprocesing_time = time.time()-start_time
vectors = [simpleVectorization(s) for s in spectrums]
index = create_IndexIVFPQR(vectors,4)
vectors_array = reshape_vectors(vectors)
vectorization_time = time.time()-(preprocesing_time+start_time)
index.train(vectors_array)
index.add(vectors_array)
D, I = index.search(vectors_array[:5], 4)
comparison_time = time.time()-(preprocesing_time+start_time+vectorization_time)
print(D)
print(I)
visualization_time = time.time()-(preprocesing_time+start_time+vectorization_time+comparison_time)
export_benchmarking("IVFPQR",preprocesing_time,comparison_time,visualization_time,vectorization_time)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
