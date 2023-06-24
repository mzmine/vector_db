from matchms.importing import load_from_mgf
import gensim
from ms2deepscore.models import load_model
def loadScpectrums(path):
    spectrums = list(load_from_mgf(path))
    return spectrums

def importSpec2VecModel(file_path):
    model = gensim.models.Word2Vec.load(file_path)
    return model

def importMS2DeepscoreModel(file_path):
    model = load_model(file_path)
    return model