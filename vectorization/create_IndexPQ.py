import faiss


def create_IndexPQ(vectors_array):
    index = faiss.IndexPQ(len(vectors_array[0]) ,len(vectors_array[0]) ,2)
    index.train(vectors_array)
    index.add(vectors_array)
    return index