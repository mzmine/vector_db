import faiss


def create_IndexFlatIP(vectors):
    max_vector_length = max(v.shape[1] for v in vectors)
    index = faiss.IndexFlatIP(max_vector_length * 2)
    return index