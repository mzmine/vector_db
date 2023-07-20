import faiss
from vectorization.create_IndexScalarQuantizer import create_IndexScalarQuantizer

def create_IndexIVFScalarQuantizer(vectors, nlist, quantizer_type=faiss.ScalarQuantizer.QT_8bit):
    quantizer= create_IndexScalarQuantizer(vectors, quantizer_type)
    max_vector_length = max(v.shape[1] for v in vectors)
    index = faiss.IndexIVFScalarQuantizer(quantizer, max_vector_length*2, nlist, quantizer_type)
    return index