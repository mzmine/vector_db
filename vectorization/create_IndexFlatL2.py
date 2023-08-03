import faiss

def create_IndexFlatL2(vectors_array):
    index = faiss.IndexFlatL2(len(vectors_array[0]))
    index.add(vectors_array)
    return index