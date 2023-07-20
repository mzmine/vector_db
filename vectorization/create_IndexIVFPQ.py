import faiss
from vectorization.create_IndexPQ import create_IndexPQ

def create_IndexIVFPQ(vectors, nlist):
    quantizer= create_IndexPQ(vectors)
    max_vector_length = max(v.shape[1] for v in vectors)
    index = faiss.IndexIVFPQ(quantizer,max_vector_length * 2, nlist, max_vector_length * 2,2)
    return index