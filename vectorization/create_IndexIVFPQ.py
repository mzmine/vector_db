import faiss
from vectorization.create_IndexPQ import create_IndexPQ

def create_IndexIVFPQ(vectors_array, nlist):
    quantizer= create_IndexPQ(vectors_array)
    index = faiss.IndexIVFPQ(quantizer,len(vectors_array[0]), nlist, len(vectors_array[0]),2)
    index.train(vectors_array)
    index.add(vectors_array)
    return index