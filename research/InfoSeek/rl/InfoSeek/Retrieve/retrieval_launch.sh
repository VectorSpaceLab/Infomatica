export CUDA_VISIBLE_DEVICES=4,5,6,7


index_file=path/to/downloaded/index/m3_Flat_512.index
corpus_file=path/to/downloaded/wiki/file/wiki-25-512.jsonl
retriever_name=m3
retriever_path=BAAI/bge-m3
pooling_method=cls

python retrieval_server.py --index_path ${index_file} \
                                            --corpus_path ${corpus_file} \
                                            --topk 3 \
                                            --retriever_name ${retriever_name} \
                                            --retriever_model ${retriever_path} \
                                            --faiss_gpu \
                                            --pooling ${pooling_method}
