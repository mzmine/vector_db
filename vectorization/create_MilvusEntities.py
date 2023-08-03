
def create_MilvusEntities(vectors_array):
    entities = [
        [i for i in range(len(vectors_array))],  # field pk
        [v for v in vectors_array],  # field embeddings
    ]
    return entities