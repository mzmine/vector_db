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
from vectorization.create_MilvusEntities import create_MilvusEntities
from vectorization.create_MilvusCollection import create_MilvusCollection
from vectorization.create_MilvusIndexIVFFlatL2 import create_MilvusIndexIVFFlatL2
from vectorization.create_MilvusIndexFlatL2 import create_MilvusIndexFlatL2
from comparison.cosine_greedy import cosine_greedy
from comparison.modified_cosine import modified_cosine
from comparison.spec2vec import spec2vec
from comparison.ms2deepscore import ms2deepscore
from visualization.plot_scores_restrictive import plotScoresRestrictive
from visualization.plot_scores import plotScores
from visualization.get_best_matches import getBestMatches
from statistics.export_benchmarking import export_benchmarking
import time


from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
import random

"""
start_time = time.time()
"""
spectrums= loadScpectrums("C:\\Users\\usuario\\Desktop\\GSOC\\vector_db\\GNPS-NIH-NATURALPRODUCTSLIBRARY.mgf")
spectrums = [metadata_processing(s) for s in spectrums]
spectrums = [peak_processing(s) for s in spectrums]
"""
preprocesing_time = time.time()-start_time
"""
vectors = [simpleVectorization(s) for s in spectrums]
vectors_array = reshape_vectors(vectors)
"""
index = create_IndexIVFPQR(vectors_array,4)
vectorization_time = time.time()-(preprocesing_time+start_time)
D, I = index.search(vectors_array[:5], 4)
comparison_time = time.time()-(preprocesing_time+start_time+vectorization_time)
print(D)
print(I)
visualization_time = time.time()-(preprocesing_time+start_time+vectorization_time+comparison_time)
export_benchmarking("IVFPQR",preprocesing_time,comparison_time,visualization_time,vectorization_time)
"""

connections.connect("default", host="localhost", port="19530")
entities= create_MilvusEntities(vectors_array)
milvus_vectors=create_MilvusCollection(vectors_array,entities)
milvus_vectors=create_MilvusIndexFlatL2(milvus_vectors)
vectors_to_search = entities[-1][2:]
search_params = {
    "metric_type": "L2",
    #"params": {"nprobe": 10},
}
result = milvus_vectors.search(vectors_to_search, "embeddings", search_params, limit=3, output_fields=["pk"])
print(result)
result = milvus_vectors.search(vectors_to_search, "embeddings", search_params, limit=3, expr="pk > 10", output_fields=["pk"])
print(result)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
