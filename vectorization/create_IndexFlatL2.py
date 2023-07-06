import faiss

def create_IndexFlatL2(vectors):
    max_vector_length = max(v.shape[1] for v in vectors)
    index = faiss.IndexFlatL2(max_vector_length * 2)
    return index