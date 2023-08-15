def create_MilvusIndexIVFPQ(milvus_vectors,metric_type):
    index = {
        "index_type": "IVF_PQ",
        "metric_type": metric_type,
        "params": {"nlist": 128,
                   "m": 445},
    }
    milvus_vectors.create_index("embeddings", index)
    milvus_vectors.load()
    return milvus_vectors