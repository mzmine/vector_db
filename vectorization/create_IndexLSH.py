import faiss


def create_IndexLSH (vectors):
    max_vector_length = max(v.shape[1] for v in vectors)
    index = faiss.IndexLSH(max_vector_length * 2,max_vector_length * 2)
    return index