def create_MilvusIndexIVFSQ8(milvus_vectors,metric_type):
    index = {
        "index_type": "IVF_SQ8",
        "metric_type": metric_type,
        "params": {"nlist": 128},
    }
    milvus_vectors.create_index("embeddings", index)
    milvus_vectors.load()
    return milvus_vectors