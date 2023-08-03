import faiss


def create_IndexScalarQuantizer(vectors_array, quantizer_type=faiss.ScalarQuantizer.QT_8bit):
    index = faiss.IndexScalarQuantizer(len(vectors_array[0]),quantizer_type)
    index.train(vectors_array)
    index.add(vectors_array)
    return index