
def create_MilvusIndexFlat(milvus_vectors,metric_type):
    index = {
        "index_type": "FLAT",
        "metric_type": metric_type,
    }
    milvus_vectors.create_index("embeddings", index)
    milvus_vectors.load()
    return milvus_vectors