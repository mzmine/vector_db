
from spec2vec import SpectrumDocument
def get_Spec2Vec_vectors(model):
    wv_array= model.wv
    vectors_array=wv_array.vectors
    return vectors_array