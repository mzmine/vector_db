from vectorization.create_IndexFlatL2 import create_IndexFlatL2
import faiss

def create_IndexIVFFlat(vectors_array, ncells, nprobe=1):
    quantizer = create_IndexFlatL2(vectors_array)
    index = faiss.IndexIVFFlat(quantizer,len(vectors_array[0]),ncells)
    index.nprobe=nprobe
    index.train(vectors_array)
    index.add(vectors_array)
    return index