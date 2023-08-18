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
from preprocessing.binPeakList import bin_peak_list
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
from vectorization.simple_vectorization2 import simpleVectorization2
from vectorization.create_MilvusEntities import create_MilvusEntities
from vectorization.create_MilvusCollection import create_MilvusCollection
from vectorization.create_MilvusIndexIVFFlat import create_MilvusIndexIVFFlat
from vectorization.create_MilvusIVFSP import create_MilvusIVFSP
from vectorization.create_MilvusIndexHNSW import create_MilvusIndexHNSW
from vectorization.create_MilvusIndexANNOY import create_MilvusIndexANNOY
from vectorization.create_MilvusIndexIVFSQ8 import create_MilvusIndexIVFSQ8
from vectorization.create_MilvusIndexFlat import create_MilvusIndexFlat
from vectorization.create_MilvusFlatSP import create_MilvusFlatSP
from vectorization.create_MilvusIndexIVFPQ import create_MilvusIndexIVFPQ
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
import logging
import pyteomics.mgf
from tqdm import tqdm
import pandas as pd


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def remove_random(peaks, remove_n=2):
    remove_indexes = np.random.randint(0, len(peaks[0]), remove_n)
    peaks = np.delete(peaks, remove_indexes, 1)
    peaks[1] = peaks[1] * (1.15 - 0.30 * np.random.random_sample(len(peaks[0])))
    return peaks


def read_mgf(input_file, min_signals=4, max_mz=1500, min_rel_intensity=0.001):
    spectra = []
    libids = []
    precursors = []
    with pyteomics.mgf.MGF(input_file) as f_in:
        for spectrum_dict in tqdm(f_in):
            try:
                precursor_mz = float(spectrum_dict["params"]["pepmass"][0])
                if precursor_mz <= max_mz and len(spectrum_dict["intensity array"]) > 0 and \
                        int(spectrum_dict["params"]["libraryquality"]) <= 3 and spectrum_dict["params"]["ionmode"] == \
                        "Positive":
                    threshold = min_rel_intensity * max(spectrum_dict["intensity array"])
                    peaks = np.array([(mz, intensity) for mz, intensity in zip(spectrum_dict["m/z array"], spectrum_dict[
                        "intensity array"]) if intensity >= threshold])

                    sum_by_max = peaks[1].sum() / peaks[1].max()
                    print(sum_by_max)
                    if (peaks.shape[0] >= min_signals) and (sum_by_max > 1):
                        spectra.append(np.array(peaks, dtype="float32").T)
                        libids.append(spectrum_dict["params"].get("spectrumid", None))
                        precursors.append(precursor_mz)
                        if len(precursors)>10000:
                            break
            except:
                # logger.warning("Cannot read spectrum "+str(spectrum_dict))
                pass

    return pd.DataFrame(
        {
            "gnpsid": libids,
            "precursor_mz": precursors,
            "peaks": spectra
        }
    )


def argsort(*arrays):
    indexes = arrays[0].argsort()
    return tuple((a[indexes] for a in arrays))

start_time = time.time()
# lib1 = r"D:\Data\lib\BILELIB19.mgf"
lib1 = r"C:\Users\migue\OneDrive\Escritorio\GSOC\vector_db\GNPS-NIH-NATURALPRODUCTSLIBRARY.mgf"
libname = "gnps"

#  use 11 as min mz as we are also using it for neutral losses
min_mz = 11
max_mz = 1500
logger.info("Read mgf")
spec_df = read_mgf(lib1, 4, max_mz, 0.001)
logger.info("read done; create vectors for {}".format(spec_df.shape[0]))
spec_df["npeaks"] = [len(peaks[1]) for peaks in spec_df["peaks"]]
spec_df["max_i"] = [peaks[1].max() for peaks in spec_df["peaks"]]
spec_df["sum_i"] = [peaks[1].sum() for peaks in spec_df["peaks"]]
spec_df["sum_by_max"] = spec_df["sum_i"] / spec_df["max_i"]
spec_df = spec_df.sort_values(by="sum_by_max", ascending=False).reset_index()
spec_df.to_csv(f"{libname}_calc.csv")

preprocesing_time = time.time()-start_time

vectors_array = np.array([bin_peak_list(peaks, min_mz, max_mz, 0.05, precursor, include_neutral_loss=True) for
                       precursor, peaks in zip(spec_df["precursor_mz"], spec_df["peaks"])], dtype="float32")
index= create_IndexFlatL2(vectors_array)
vectorization_time = time.time()-(preprocesing_time+start_time)
D, I = index.search(vectors_array[:5], 4)
comparison_time = time.time()-(preprocesing_time+start_time+vectorization_time)
print(D)
print(I)
visualization_time = time.time()-(preprocesing_time+start_time+vectorization_time+comparison_time)
export_benchmarking("FlatL2",preprocesing_time,comparison_time,visualization_time,vectorization_time)





# See PyCharm help at https://www.jetbrains.com/help/pycharm/
