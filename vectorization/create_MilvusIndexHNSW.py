def create_MilvusIndexHNSW(milvus_vectors,metric_type):
    index = {
        "index_type": "HNSW",
        "metric_type": metric_type,
        "M": 8,
        "efConstruction":16
    }
    milvus_vectors.create_index("embeddings", index)
    milvus_vectors.load()
    return milvus_vectors
