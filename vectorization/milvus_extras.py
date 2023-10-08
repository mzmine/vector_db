from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)


def create_MilvusCollection(vectors_array, entities):
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=len(vectors_array[0]))
    ]
    schema = CollectionSchema(fields, "vector collection5")
    milvusVectors = Collection("vector_collection5", schema)

    milvusVectors.insert(entities)
    milvusVectors.flush()

    return milvusVectors

def create_MilvusEntities(vectors_array):
    entities = [
        [i for i in range(len(vectors_array))],  # field pk
        [v for v in vectors_array],  # field embeddings
    ]
    return entities

def create_MilvusFlatSP(metric_type):
    search_params = {
        "metric_type": metric_type,
    }
    return search_params

def create_MilvusIVFSP(metric_type, nprobe):
    search_params = {
        "metric_type": metric_type,
        "nprobe": nprobe
    }
    return search_params