def create_MilvusIndexIVFFlat(milvus_vectors,metric_type):
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": metric_type,
        "params": {"nlist": 128},
    }
    milvus_vectors.create_index("embeddings", index)
    milvus_vectors.load()
    return milvus_vectors