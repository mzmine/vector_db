import faiss
from vectorization.create_IndexPQ import create_IndexPQ

def create_IndexIVFPQR(vectors, nlist):
    quantizer= create_IndexPQ(vectors)
    max_vector_length = max(v.shape[1] for v in vectors)
    index = faiss.IndexIVFPQR(quantizer,max_vector_length * 2, nlist, 8,8,8,8)
    return index