def create_MilvusIndexIVFFlatL2(milvus_vectors):
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 128},
    }
    milvus_vectors.create_index("embeddings", index)
    milvus_vectors.load()
    return milvus_vectors