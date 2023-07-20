import faiss

def create_IndexHNSWFlat(vectors, M):
    max_vector_length = max(v.shape[1] for v in vectors)
    index = faiss.IndexHNSWFlat(max_vector_length * 2, M)
    return index