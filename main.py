




import numpy as np
import sys
from IO.export import generateNetwork
from IO.importing import loadScpectrums
from IO.importing import importSpec2VecModel
from IO.importing import importMS2DeepscoreModel
from IO.importing import read_mgf
from IO.importing import loadSpectraBlink
from preprocessing.metadata_processing import metadata_processing
from preprocessing.peak_processing import peak_processing
from preprocessing.train_spec2vec import train_spec2vec
from preprocessing.binPeakList import bin_peak_list
from preprocessing.get_Spec2Vec_vectors import get_Spec2Vec_vectors
from preprocessing.get_MS2DeepScore_vectors import get_MS2DeepScore_vectors
from vectorization.reshape_vectors import reshape_vectors
from vectorization.faissIndexes import create_IndexFlatL2
from vectorization.faissIndexes import create_IndexFlatIP
from vectorization.faissIndexes import create_IndexIVFFlat
from vectorization.faissIndexes import create_IndexLSH
from vectorization.faissIndexes import create_IndexHNSWFlat
from vectorization.faissIndexes import create_IndexIVFScalarQuantizer
from vectorization.faissIndexes import create_IndexIVFPQ
from vectorization.faissIndexes import create_IndexIVFPQR
from vectorization.faissIndexes import create_IndexPQ
from vectorization.faissIndexes import create_IndexScalarQuantizer
from vectorization.simple_vectorization import simple_vectorization
from vectorization.simple_vectorization2 import simple_vectorization2
from vectorization.milvus_extras import create_MilvusEntities
from vectorization.milvus_extras import create_MilvusCollection
from vectorization.milvusIndexes import create_MilvusIndexIVFFlat
from vectorization.milvus_extras import create_MilvusIVFSP
from vectorization.milvusIndexes import create_MilvusIndexHNSW
from vectorization.milvusIndexes import create_MilvusIndexANNOY
from vectorization.milvusIndexes import create_MilvusIndexIVFSQ8
from vectorization.milvusIndexes import create_MilvusIndexFlat
from vectorization.milvus_extras import create_MilvusFlatSP
from vectorization.milvusIndexes import create_MilvusIndexIVFPQ
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
)
import logging
import yaml
import subprocess
import os


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
with open('indexesConfig.yaml', 'r') as file:
    yaml_data = yaml.safe_load(file)

def remove_random(peaks, remove_n=2):
    remove_indexes = np.random.randint(0, len(peaks[0]), remove_n)
    peaks = np.delete(peaks, remove_indexes, 1)
    peaks[1] = peaks[1] * (1.15 - 0.30 * np.random.random_sample(len(peaks[0])))
    return peaks
def argsort(*arrays):
    indexes = arrays[0].argsort()
    return tuple((a[indexes] for a in arrays))

def indexSearch(vectors_array, yaml_data):
    index_type = yaml_data['index']
    if yaml_data['mode'] == "faiss":
        if index_type in yaml_data['faiss_indexes']:
            function_name = yaml_data['faiss_indexes'][index_type]['function']
            if function_name in globals():
                selected_function = globals()[function_name]
            if 'param3' in yaml_data['faiss_indexes'][index_type]:
                param1 = yaml_data['faiss_indexes'][index_type]['param1']
                param2 = yaml_data['faiss_indexes'][index_type]['param2']
                param3 = yaml_data['faiss_indexes'][index_type]['param3']
                index = selected_function(vectors_array, param1, param2, param3)
            elif 'param2' in yaml_data['faiss_indexes'][index_type]:
                param1 = yaml_data['faiss_indexes'][index_type]['param1']
                param2 = yaml_data['faiss_indexes'][index_type]['param2']
                index = selected_function(vectors_array, param1, param2)
            elif 'param1' in yaml_data['faiss_indexes'][index_type]:
                param1 = yaml_data['faiss_indexes'][index_type]['param1']
                index = selected_function(vectors_array, param1)
            else:
                index = selected_function(vectors_array)
        D, I = index.search(vectors_array[:5], 4)
        print(D)
        print(I)
    elif yaml_data['mode'] == "milvus":
        search_type = yaml_data['search_params_type']
        command = ["sudo", "docker-compose", "up", "-d"]
        subprocess.run(command)
        time.sleep(60)
        connections.connect("default", host="localhost", port="19530")
        entities = create_MilvusEntities(vectors_array)
        milvus_vectors = create_MilvusCollection(vectors_array, entities)
        if search_type in yaml_data['search_parameters']:
            function_name = yaml_data['search_parameters'][search_type]['function']
            metric_type = yaml_data['metric_type']
            if function_name in globals():
                selected_function = globals()[function_name]

            if 'param1' in yaml_data['search_parameters'][search_type]:
                param1 = yaml_data['search_parameters'][search_type]['param1']
                search_params = selected_function(metric_type, param1)
            else:
                search_params = selected_function(metric_type)
            function_name = yaml_data['milvus_indexes'][index_type]['function']
            if function_name in globals():
                selected_function2 = globals()[function_name]
            if 'param2' in yaml_data['milvus_indexes'][index_type]:
                param2 = yaml_data['milvus_indexes'][index_type]['param2']
                param1 = yaml_data['milvus_indexes'][index_type]['param1']
                milvus_vectors = selected_function2(milvus_vectors, metric_type, param1, param2)
            elif 'param1' in yaml_data['milvus_indexes'][index_type]:
                param1 = yaml_data['milvus_indexes'][index_type]['param1']
                milvus_vectors = selected_function2(milvus_vectors, metric_type, param1)
            else:
                milvus_vectors = selected_function2(milvus_vectors, metric_type)
            milvus_vectors.load()
            vectors_to_search = entities[-1][-2:]
            result = milvus_vectors.search(vectors_to_search, "embeddings", search_params, limit=6,
                                           output_fields=["pk"])
            milvus_vectors.release()
            milvus_vectors.drop_index()
            return result

start_time = time.time()
lib1 = os.path.abspath('GNPS-NIH-NATURALPRODUCTSLIBRARY.mgf')
model_path = os.path.abspath('MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5')
libname = "gnps"
model_file= "New model3"
importing = sys.argv[1]
preprocessing = sys.argv[2]
vectorization = sys.argv[3]
if importing == 'matchms':
    spectra = loadScpectrums(lib1)
elif importing == 'blink':
    spectra = loadSpectraBlink(lib1)
elif importing == 'importR':
    spectra = read_mgf(lib1)
if preprocessing == 'normal':
    spectra = [peak_processing(spectrum) for spectrum in spectra]
    spectra = [metadata_processing(spectrum) for spectrum in spectra]
if vectorization == 'simple':
    vectors = [simple_vectorization(spectrum) for spectrum in spectra]
    vectors_array = reshape_vectors(vectors)
elif vectorization == 'simple2':
    vectors_array = [simple_vectorization2(spectrum) for spectrum in spectra]
elif vectorization == 'Ms2DeepScore':
    model = importMS2DeepscoreModel(model_path)
    vectors_array = get_MS2DeepScore_vectors(spectra,model)
elif vectorization == 'Spec2Vec':
    model = train_spec2vec(model_file, spectra)
    vectors_array = get_Spec2Vec_vectors(model)



#vectors_array = np.array([bin_peak_list(peaks, min_mz, max_mz, 0.05, precursor, include_neutral_loss=True) for
                       #precursor, peaks in zip(spec_df["precursor_mz"], spec_df["peaks"])], dtype="float32")
result= indexSearch(vectors_array,yaml_data)
print(result)
command = ["sudo", "docker-compose", "down", "--volumes"]
subprocess.run(command)



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
