import faiss
from vectorization.create_IndexScalarQuantizer import create_IndexScalarQuantizer

def create_IndexIVFScalarQuantizer(vectors_array, nlist, quantizer_type=faiss.ScalarQuantizer.QT_8bit):
    quantizer= create_IndexScalarQuantizer(vectors_array, quantizer_type)
    index = faiss.IndexIVFScalarQuantizer(quantizer, len(vectors_array[0]), nlist, quantizer_type)
    index.train(vectors_array)
    index.add(vectors_array)
    return index