from pymilvus import (
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

def create_MilvusCollection(vectors_array,entities):
    fields = [
        FieldSchema(name="pk", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=len(vectors_array[0]))
    ]
    schema = CollectionSchema(fields, "vector collection")
    milvusVectors = Collection("vector_collection", schema)

    milvusVectors.insert(entities)
    milvusVectors.flush()
    
    return milvusVectors