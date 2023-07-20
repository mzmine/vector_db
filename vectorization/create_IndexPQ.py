import faiss


def create_IndexPQ(vectors):
    max_vector_length = max(v.shape[1] for v in vectors)
    index = faiss.IndexPQ(max_vector_length * 2,max_vector_length * 2,2)
    return index