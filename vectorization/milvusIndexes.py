def create_MilvusIndexANNOY(milvus_vectors,metric_type,n_trees):
    index = {
        "index_type": "ANNOY",
        "metric_type": metric_type,
        "n_trees": n_trees
    }
    milvus_vectors.create_index("embeddings", index)
    milvus_vectors.load()
    return milvus_vectors

def create_MilvusIndexFlat(milvus_vectors,metric_type):
    index = {
        "index_type": "FLAT",
        "metric_type": metric_type,
    }
    milvus_vectors.create_index("embeddings", index)
    milvus_vectors.load()
    return milvus_vectors

def create_MilvusIndexHNSW(milvus_vectors,metric_type,M,efConstrucion):
    index = {
        "index_type": "HNSW",
        "metric_type": metric_type,
        "M": M,
        "efConstruction":efConstrucion
    }
    milvus_vectors.create_index("embeddings", index)
    milvus_vectors.load()
    return milvus_vectors

def create_MilvusIndexIVFFlat(milvus_vectors,metric_type,nlist):
    index = {
        "index_type": "IVF_FLAT",
        "metric_type": metric_type,
        "params": {"nlist": nlist},
    }
    milvus_vectors.create_index("embeddings", index)
    milvus_vectors.load()
    return milvus_vectors

def create_MilvusIndexIVFPQ(milvus_vectors,metric_type, nlist, m):
    index = {
        "index_type": "IVF_PQ",
        "metric_type": metric_type,
        "params": {"nlist": nlist,
                   "m": m},
    }
    milvus_vectors.create_index("embeddings", index)
    milvus_vectors.load()
    return milvus_vectors

def create_MilvusIndexIVFSQ8(milvus_vectors,metric_type):
    index = {
        "index_type": "IVF_SQ8",
        "metric_type": metric_type,
        "params": {"nlist": 128},
    }
    milvus_vectors.create_index("embeddings", index)
    milvus_vectors.load()
    return milvus_vectors