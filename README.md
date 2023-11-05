# vector_db
# Introduction
The goal of this repository is to create a vector database joining the functionality of different vector comparison programs such as matchms, Ms2DeepScore, faiss or milvus in order to provide the variability needed to obtain the optimal search results of the query vectors.  
This is oriented to the search of the nearest neighbour of molecules in the field of proteomics.
# Structure
The code is divided in 4 main processes:  
-  IO  
-  Preprocessing
-  Vectorization
-  Index search
## IO 
The main purpose of this process is the import/export of the data used in the program.  
In the importing section there are two possible methods to import data:  
-  Matchms
-  Blink*
-  ImportR*
     
The results of the search are printed on the console and a representation of the nearest search molecules is exported to the working directory.  
Milvus:
![image](https://github.com/miguelroblesuriel/vector_db/assets/120097028/cb8115cf-60e5-4a90-ba13-37abb61a7168)
Faiss(First matrix distances, second matrix ids):  
![image](https://github.com/miguelroblesuriel/vector_db/assets/120097028/1999df96-58e7-40ab-851a-7e5772446e8a)  
Representation of molecules:  
Compound0:  
![compound_0](https://github.com/miguelroblesuriel/vector_db/assets/120097028/039588c8-aafb-4049-baa1-e8317ba5df94)  
Compounds 782, 966, 487:  
![compound_782](https://github.com/miguelroblesuriel/vector_db/assets/120097028/d53c982c-77c2-4b52-80e9-4c5878642a26)
![compound_966](https://github.com/miguelroblesuriel/vector_db/assets/120097028/04797ac3-978e-4637-addb-c2d1af40021e)
![compound_487](https://github.com/miguelroblesuriel/vector_db/assets/120097028/3ebafbc3-02d4-498c-8670-94f8fc7aea6a)







## Preprocessing 
During this process the imported spectra is preprocessed in order to be managed by the next process, currently there is only one way of preprocessing which is by using the methods given by matchms.
## Vectorization
This processes is used to transform the data of the preprocessed spectra into vectors containig the information needed in order to do the search, the program currently provides 4 possible ways of vectorizing spectra:
-  Simple
-  Simple2
-  Ms2DeepScore
-  Spec2Vec
  
Simple and Simple2 are original methods of the program while Ms2DeepScore and Spec2Vec makes use of the methods created by this two libraries to vectorize spectra.
## Index search
The last process consist on the creation of different indexes, either of faiss or milvus, to made the search in our database and provide the nearest neighbours of our query vectors.  
The program supports different type of indexes which can be clasified by this manner:
- Faiss
  - flatIP
  - flatL2
  - HNSWFlat*
  - IVFFlat
  - IVFPQ
  - IVFPQR
  - IVFScalarQuantizer
  - LSH
  - PQ
  - ScalarQuantizer
- Milvus
  - Indexes
    - ANNOY
    - Flat
    - HSNW
    - IVFFlat
    - IVFPQ
    - IVFSQ8
  - Search Parameters
    - flat
    - IVF
# How to use
In order to execute the program I have been using the WSL console to support the docker container needed to execute the milvus indexes. To run the program tou would need to enter the following command line: "python3 main.py [import method] [preprocessing method] [vectorization method]. Currently not all the importing methods can be used due to data type incompatiblities that need to be solved, this with the fact that there is only 1 preprocessing method at the moment leave the command as "python3 main.py matchms normal [vectorization method]".  
The command line of the console can only modify the first three processes, the index search process can be modified changing the indexconfig.yaml file of the repository. In this file you can choose between faiss or milvus indexes by changing the name of the program and index or search parameters to use, you can also modify the internal index parameters in order to optimize the of your specific vector search.
Finally the file with the spectra is selected within the code in the lib1 variable and the query vectors are also selected within the code in the vectors_to_search variable.
  


  
