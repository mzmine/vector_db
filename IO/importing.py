from matchms.importing import load_from_mgf
import gensim
from ms2deepscore.models import load_model
import numpy as np
import pyteomics.mgf
from tqdm import tqdm
import pandas as pd
import blink
import sys

from ftplib import FTP
def loadScpectrums(path):
    spectrums = list(load_from_mgf(path))
    return spectrums

def importSpec2VecModel(file_path):
    model = gensim.models.Word2Vec.load(file_path)
    return model

def importMS2DeepscoreModel(file_path):
    model = load_model(file_path)
    return model

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

def loadSpectraBlink(file):
    mgf = blink.open_msms_file(file)
    discretized_spectra = blink.discretize_spectra(mgf.spectrum.tolist(),
                                                   mgf.precursor_mz.tolist(),
                                                   bin_width=0.001, tolerance=0.01, intensity_power=0.5,
                                                   trim_empty=False, remove_duplicates=False, network_score=False)

    return discretized_spectra
