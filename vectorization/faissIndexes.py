import faiss


def create_IndexFlatIP(vectors_array):
    index = faiss.IndexFlatIP(len(vectors_array[0]))
    index.add(vectors_array)
    return index

def create_IndexFlatL2(vectors_array):
    index = faiss.IndexFlatL2(len(vectors_array[0]))
    index.add(vectors_array)
    return index

def create_IndexHNSWFlat(vectors_array, M):
    index = faiss.IndexHNSWFlat(len(vectors_array[0]), M)
    return index

def create_IndexIVFFlat(vectors_array, nlists, nprobe=1):
    quantizer = create_IndexFlatL2(vectors_array)
    index = faiss.IndexIVFFlat(quantizer,len(vectors_array[0]),nlists)
    index.nprobe=nprobe
    index.train(vectors_array)
    index.add(vectors_array)
    return index

def create_IndexIVFPQ(vectors_array, nlist, M, nbits):
    quantizer= create_IndexPQ(vectors_array, M, nbits)
    index = faiss.IndexIVFPQ(quantizer,len(vectors_array[0]), nlist, M,nbits)
    index.train(vectors_array)
    index.add(vectors_array)
    return index

def create_IndexIVFPQR(vectors_array, nlist, M, nbits):
    quantizer= create_IndexPQ(vectors_array,M,nbits)
    training_points = 100 * nlist
    quantizer.train(vectors_array[:training_points])
    index = faiss.IndexIVFPQR(quantizer,len(vectors_array[0]), nlist, M,nbits,len(vectors_array[0]),4)
    index.train(vectors_array)
    return index

def create_IndexIVFScalarQuantizer(vectors_array, nlist, quantizer_type=faiss.ScalarQuantizer.QT_8bit):
    quantizer= create_IndexScalarQuantizer(vectors_array, quantizer_type)
    index = faiss.IndexIVFScalarQuantizer(quantizer, len(vectors_array[0]), nlist, quantizer_type)
    index.train(vectors_array)
    index.add(vectors_array)
    return index

def create_IndexLSH (vectors_array, nbits):
    index = faiss.IndexLSH(len(vectors_array[0]),nbits)
    index.add(vectors_array)
    return index

def create_IndexPQ(vectors_array,M,nbits):
    index = faiss.IndexPQ(len(vectors_array[0]) ,M ,nbits)
    index.train(vectors_array)
    index.add(vectors_array)
    return index

def create_IndexScalarQuantizer(vectors_array, quantizer_type=faiss.ScalarQuantizer.QT_8bit):
    index = faiss.IndexScalarQuantizer(len(vectors_array[0]),quantizer_type)
    index.train(vectors_array)
    index.add(vectors_array)
    return index