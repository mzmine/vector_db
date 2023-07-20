from vectorization.create_IndexFlatL2 import create_IndexFlatL2
import faiss

def create_IndexIVFFlat(vectors, ncells, nprobe=1):
    quantizer = create_IndexFlatL2(vectors)
    max_vector_length = max(v.shape[1] for v in vectors)
    index = faiss.IndexIVFFlat(quantizer,max_vector_length*2,ncells)
    index.nprobe=nprobe
    return index