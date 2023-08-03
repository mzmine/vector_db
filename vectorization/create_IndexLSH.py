import faiss


def create_IndexLSH (vectors_array):
    index = faiss.IndexLSH(len(vectors_array[0]),len(vectors_array[0]))
    index.add(vectors_array)
    return index