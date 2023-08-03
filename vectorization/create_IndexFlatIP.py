import faiss


def create_IndexFlatIP(vectors_array):
    index = faiss.IndexFlatIP(len(vectors_array[0]))
    index.add(vectors_array)
    return index