
from spec2vec import SpectrumDocument
from spec2vec.model_building import train_new_word2vec_model


def train_spec2vec(model_file, spectrums):
    spectrum_documents = [SpectrumDocument(s, n_decimals=2) for s in spectrums]
    model = train_new_word2vec_model(spectrum_documents, iterations=[25], filename=model_file, workers=2,
                                     progress_logger=True)
    return model