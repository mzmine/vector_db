import faiss

def create_IndexHNSWFlat(vectors_array, M):
    index = faiss.IndexHNSWFlat(len(vectors_array[0]), M)
    return index