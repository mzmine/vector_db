import faiss
from vectorization.create_IndexPQ import create_IndexPQ

def create_IndexIVFPQR(vectors_array, nlist):
    quantizer= create_IndexPQ(vectors_array)
    training_points = 100 * nlist
    quantizer.train(vectors_array[:training_points])
    index = faiss.IndexIVFPQR(quantizer,len(vectors_array[0]), nlist, len(vectors_array[0]),4,len(vectors_array[0]),4)
    return index