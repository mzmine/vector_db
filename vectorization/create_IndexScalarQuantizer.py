import faiss


def create_IndexScalarQuantizer(vectors, quantizer_type=faiss.ScalarQuantizer.QT_8bit):
    max_vector_length = max(v.shape[1] for v in vectors)
    index = faiss.IndexScalarQuantizer(max_vector_length * 2,quantizer_type)
    return index