def create_MilvusIndexANNOY(milvus_vectors,metric_type,n_trees):
    index = {
        "index_type": "ANNOY",
        "metric_type": metric_type,
        "n_trees": n_trees
    }
    milvus_vectors.create_index("embeddings", index)
    milvus_vectors.load()
    return milvus_vectors
