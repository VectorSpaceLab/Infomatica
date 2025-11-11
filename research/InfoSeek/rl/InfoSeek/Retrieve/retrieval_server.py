# retrieval_server.py

import json
import os
import warnings
from typing import List, Dict, Optional
import argparse

import faiss
import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModel
from tqdm import tqdm
import datasets

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


MAX_DOC_TOKENS = 512
# NOTE: Replace with the path to your tokenizer model, e.g., a model from Hugging Face Hub.
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

def truncate_text(text: str) -> str:
    """
    Truncates text to MAX_DOC_TOKENS using the global tokenizer.
    
    Args:
        text: The input string to truncate.
        
    Returns:
        The truncated string.
    """
    if not isinstance(text, str):
        return text
    
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    truncated_token_ids = token_ids[:MAX_DOC_TOKENS]
    return tokenizer.decode(truncated_token_ids, skip_special_tokens=True)


def is_document_valid(doc: Dict, min_words: int = 20) -> bool:
    """
    Checks if a document is valid. A valid document must:
    1. Contain substantial text content besides a title.
    2. The word count of the text content must be at least min_words.

    Args:
        doc: A document dictionary, which may contain 'text' or 'contents' keys.
        min_words: The minimum required word count, defaults to 20.

    Returns:
        True if the document is valid, False otherwise.
    """
    if not doc:
        return False
    
    text_content = ""
    if 'text' in doc and doc.get('text'):
        text_content = doc.get('text', '').strip()
    elif 'contents' in doc and doc.get('contents'):
        # Assumes the main content follows the first line (title).
        parts = doc.get('contents', '').split('\n', 1)
        if len(parts) > 1:
            text_content = parts[1].strip()
            
    if not text_content:
        return False

    # Check if the text is long enough based on word count.
    word_count = len(text_content.split())
    return word_count >= min_words


def load_corpus(corpus_path: str):
    """Loads a corpus from a JSONL file using the datasets library."""
    corpus = datasets.load_dataset(
        'json', 
        data_files=corpus_path,
        split="train",
        num_proc=4
    )
    return corpus

def read_jsonl(file_path):
    """Reads a JSONL file line by line."""
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_docs(corpus, doc_idxs):
    """Loads specific documents from the corpus by their indices."""
    results = [corpus[int(idx)] for idx in doc_idxs]
    return results

def load_model(model_path: str, use_fp16: bool = False):
    """Loads a Hugging Face model and tokenizer."""
    model_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    model.cuda()
    if use_fp16: 
        model = model.half()
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    return model, tokenizer

def pooling(
    pooler_output,
    last_hidden_state,
    attention_mask = None,
    pooling_method = "mean"
):
    """Performs pooling on model outputs to get a single embedding."""
    if pooling_method == "mean":
        last_hidden = last_hidden_state.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]
    elif pooling_method == "cls":
        return last_hidden_state[:, 0]
    elif pooling_method == "pooler":
        return pooler_output
    else:
        raise NotImplementedError("Pooling method not implemented!")

class Encoder:
    """A wrapper for a sentence-transformer model to encode text into embeddings."""
    def __init__(self, model_name, model_path, pooling_method, max_length, use_fp16):
        self.model_name = model_name
        self.model_path = model_path
        self.pooling_method = pooling_method
        self.max_length = max_length
        self.use_fp16 = use_fp16

        self.model, self.tokenizer = load_model(model_path=model_path, use_fp16=use_fp16)
        self.model.eval()

    @torch.no_grad()
    def encode(self, query_list: List[str], is_query=True) -> np.ndarray:
        """Encodes a list of strings into numpy embeddings."""
        if isinstance(query_list, str):
            query_list = [query_list]

        # Add model-specific prefixes if needed (e.g., for E5, BGE models).
        if "e5" in self.model_name.lower():
            if is_query:
                query_list = [f"query: {query}" for query in query_list]
            else:
                query_list = [f"passage: {query}" for query in query_list]

        if "bge" in self.model_name.lower():
            if is_query:
                query_list = [f"Represent this sentence for searching relevant passages: {query}" for query in query_list]

        inputs = self.tokenizer(query_list,
                                max_length=self.max_length,
                                padding=True,
                                truncation=True,
                                return_tensors="pt"
                                )
        inputs = {k: v.cuda() for k, v in inputs.items()}

        if "T5" in type(self.model).__name__:
            # T5-based retrieval model
            decoder_input_ids = torch.zeros(
                (inputs['input_ids'].shape[0], 1), dtype=torch.long
            ).to(inputs['input_ids'].device)
            output = self.model(
                **inputs, decoder_input_ids=decoder_input_ids, return_dict=True
            )
            query_emb = output.last_hidden_state[:, 0, :]
        else:
            output = self.model(**inputs, return_dict=True)
            query_emb = pooling(output.pooler_output,
                                output.last_hidden_state,
                                inputs['attention_mask'],
                                self.pooling_method)
            if "dpr" not in self.model_name.lower():
                query_emb = torch.nn.functional.normalize(query_emb, dim=-1)

        query_emb = query_emb.detach().cpu().numpy()
        query_emb = query_emb.astype(np.float32, order="C")
        
        del inputs, output
        torch.cuda.empty_cache()

        return query_emb

class BaseRetriever:
    """Abstract base class for all retrievers."""
    def __init__(self, config):
        self.config = config
        self.retrieval_method = config.retrieval_method
        self.topk = config.retrieval_topk
        self.index_path = config.index_path
        self.corpus_path = config.corpus_path

    def _search(self, query: str, num: int, return_score: bool):
        raise NotImplementedError

    def _batch_search(self, query_list: List[str], num: int, return_score: bool):
        raise NotImplementedError

    def search(self, query: str, num: int = None, return_score: bool = False):
        return self._search(query, num, return_score)
    
    def batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        return self._batch_search(query_list, num, return_score)

class BM25Retriever(BaseRetriever):
    """BM25 retriever implemented using Pyserini."""
    def __init__(self, config):
        super().__init__(config)
        from pyserini.search.lucene import LuceneSearcher
        self.searcher = LuceneSearcher(self.index_path)
        self.contain_doc = self._check_contain_doc()
        if not self.contain_doc:
            self.corpus = load_corpus(self.corpus_path)
        self.max_process_num = 8
    
    def _check_contain_doc(self):
        return self.searcher.doc(0).raw() is not None

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        
        # Request more documents (+20 as a buffer) in case some are filtered out.
        search_k = num + 20 
        hits = self.searcher.search(query, k=search_k)

        if not hits:
            return ([], []) if return_score else []

        # Filter out invalid documents (e.g., empty content).
        filtered_hits = []
        for hit in hits:
            if self.contain_doc:
                doc_content = json.loads(self.searcher.doc(hit.docid).raw())
            else:
                # Fetch directly from the datasets object, which is more efficient.
                doc_content = self.corpus[int(hit.docid)]
            
            if is_document_valid(doc_content):
                filtered_hits.append(hit)
        
        # Select the top 'num' from the valid documents.
        final_hits = filtered_hits[:num]

        if len(final_hits) < num:
            warnings.warn(f'Could not retrieve {num} valid documents, found {len(final_hits)} instead.')
            
        scores = [hit.score for hit in final_hits]

        # Format and truncate text for the final results.
        if self.contain_doc:
            all_contents = [json.loads(self.searcher.doc(hit.docid).raw())['contents'] for hit in final_hits]
            truncated_contents = [truncate_text(content) for content in all_contents]
            results = [
                {
                    'title': content.split("\n", 1)[0].strip("\""),
                    'text': content.split("\n", 1)[1] if len(content.split("\n", 1)) > 1 else "",
                    'contents': content
                } 
                for content in truncated_contents
            ]
        else:
            results = load_docs(self.corpus, [hit.docid for hit in final_hits])
            for doc in results:
                if 'contents' in doc:
                    doc['contents'] = truncate_text(doc.get('contents'))
                elif 'text' in doc:
                    doc['text'] = truncate_text(doc.get('text'))
        
        if return_score:
            return results, scores
        else:
            return results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        results = []
        scores = []
        for query in query_list:
            item_result, item_score = self._search(query, num, True)
            results.append(item_result)
            scores.append(item_score)
        if return_score:
            return results, scores
        else:
            return results

class DenseRetriever(BaseRetriever):
    """Dense retriever using a sentence-transformer model and Faiss index."""
    def __init__(self, config):
        super().__init__(config)
        self.index = faiss.read_index(self.index_path)
        if config.faiss_gpu:
            co = faiss.GpuMultipleClonerOptions()
            co.useFloat16 = True
            co.shard = True
            self.index = faiss.index_cpu_to_all_gpus(self.index, co=co)

        self.corpus = load_corpus(self.corpus_path)
        self.encoder = Encoder(
            model_name = self.retrieval_method,
            model_path = config.retrieval_model_path,
            pooling_method = config.retrieval_pooling_method,
            max_length = config.retrieval_query_max_length,
            use_fp16 = config.retrieval_use_fp16
        )
        self.topk = config.retrieval_topk
        self.batch_size = config.retrieval_batch_size

    def _search(self, query: str, num: int = None, return_score: bool = False):
        if num is None:
            num = self.topk
        
        # Request more documents (+20 as a buffer).
        search_k = num + 20
        query_emb = self.encoder.encode(query)
        scores, idxs = self.index.search(query_emb, k=search_k)
        
        scores = scores[0]
        idxs = idxs[0]

        valid_results = []
        valid_scores = []
        
        # Filter documents.
        for idx, score in zip(idxs, scores):
            if len(valid_results) >= num:
                break  # Found enough valid documents.
            if idx == -1:  # Faiss index might return -1.
                continue
                
            doc = self.corpus[int(idx)]
            if is_document_valid(doc):
                processed_doc = doc.copy()
                # Truncate the content of the valid document.
                if 'contents' in processed_doc:
                    processed_doc['contents'] = truncate_text(processed_doc.get('contents'))
                elif 'text' in processed_doc:
                    processed_doc['text'] = truncate_text(processed_doc.get('text'))
                valid_results.append(processed_doc)
                valid_scores.append(score.item())

        if len(valid_results) < num:
            warnings.warn(f'Could not retrieve {num} valid documents, found {len(valid_results)} instead.')

        if return_score:
            return valid_results, valid_scores
        else:
            return valid_results

    def _batch_search(self, query_list: List[str], num: int = None, return_score: bool = False):
        if isinstance(query_list, str):
            query_list = [query_list]
        if num is None:
            num = self.topk
        
        # Request more documents (+20 as a buffer).
        search_k = num + 20

        results = []
        scores = []
        for start_idx in tqdm(range(0, len(query_list), self.batch_size), desc='Retrieval process: '):
            query_batch = query_list[start_idx:start_idx + self.batch_size]
            batch_size_current = len(query_batch)
            
            batch_emb = self.encoder.encode(query_batch)
            # Retrieve search_k results for each query in the batch.
            batch_scores_unfiltered, batch_idxs_unfiltered = self.index.search(batch_emb, k=search_k)
            
            # Process the results for each query in the batch.
            for i in range(batch_size_current):
                query_idxs = batch_idxs_unfiltered[i]
                query_scores = batch_scores_unfiltered[i]
                
                valid_docs_for_query = []
                valid_scores_for_query = []

                for idx, score in zip(query_idxs, query_scores):
                    if len(valid_docs_for_query) >= num:
                        break # Found enough valid documents for the current query.
                    if idx == -1:
                        continue
                    
                    doc = self.corpus[int(idx)]
                    if is_document_valid(doc):
                        processed_doc = doc.copy()
                        # Truncate the content of the valid document.
                        if 'contents' in processed_doc:
                            processed_doc['contents'] = truncate_text(processed_doc.get('contents'))
                        elif 'text' in processed_doc:
                            processed_doc['text'] = truncate_text(processed_doc.get('text'))
                        valid_docs_for_query.append(processed_doc)
                        valid_scores_for_query.append(score.item())
                
                if len(valid_docs_for_query) < num:
                    warnings.warn(f'For query "{query_batch[i][:50]}...", could not retrieve {num} valid documents, found {len(valid_docs_for_query)} instead.')

                results.append(valid_docs_for_query)
                scores.append(valid_scores_for_query)

            del batch_emb, batch_scores_unfiltered, batch_idxs_unfiltered, query_batch
            torch.cuda.empty_cache()
            
        if return_score:
            return results, scores
        else:
            return results


def get_retriever(config):
    """Factory function to instantiate the correct retriever based on config."""
    if config.retrieval_method == "bm25":
        return BM25Retriever(config)
    else:
        return DenseRetriever(config)


#####################################
#         FastAPI Server            #
#####################################

class Config:
    """
    Configuration class to hold all settings.
    In a real application, this might be loaded from a file or environment variables.
    """
    def __init__(
        self, 
        retrieval_method: str = "bm25", 
        retrieval_topk: int = 10,
        index_path: str = "./index/bm25_index",
        corpus_path: str = "./data/corpus.jsonl",
        faiss_gpu: bool = True,
        retrieval_model_path: str = "intfloat/e5-base-v2",
        retrieval_pooling_method: str = "mean",
        retrieval_query_max_length: int = 256,
        retrieval_use_fp16: bool = False,
        retrieval_batch_size: int = 128
    ):
        self.retrieval_method = retrieval_method
        self.retrieval_topk = retrieval_topk
        self.index_path = index_path
        self.corpus_path = corpus_path
        self.faiss_gpu = faiss_gpu
        self.retrieval_model_path = retrieval_model_path
        self.retrieval_pooling_method = retrieval_pooling_method
        self.retrieval_query_max_length = retrieval_query_max_length
        self.retrieval_use_fp16 = retrieval_use_fp16
        self.retrieval_batch_size = retrieval_batch_size


class QueryRequest(BaseModel):
    queries: List[str]
    topk: Optional[int] = None
    return_scores: bool = False


app = FastAPI()

@app.post("/retrieve")
def retrieve_endpoint(request: QueryRequest):
    """
    API endpoint to perform retrieval for a batch of queries.
    
    Example request body:
    {
      "queries": ["What is Python?", "Tell me about neural networks."],
      "topk": 3,
      "return_scores": true
    }
    """
    if not request.topk:
        request.topk = config.retrieval_topk  # Fallback to default topk

    # Perform batch retrieval.
    results, scores = retriever.batch_search(
        query_list=request.queries,
        num=request.topk,
        return_score=True  # Always get scores to simplify response formatting.
    )
    
    # Format the response.
    resp = []
    for i, single_result in enumerate(results):
        if request.return_scores:
            # Combine documents and scores if requested.
            combined = []
            for doc, score in zip(single_result, scores[i]):
                combined.append({"document": doc, "score": score})
            resp.append(combined)
        else:
            # Otherwise, just return the documents.
            resp.append(single_result)
            
    return {"result": resp}


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Launch the retrieval server.")
    parser.add_argument("--index_path", type=str, default="path/to/your/index_file", help="Path to the corpus index file (Faiss or Pyserini).")
    parser.add_argument("--corpus_path", type=str, default="path/to/your/corpus.jsonl", help="Path to the corpus file in JSONL format.")
    parser.add_argument("--topk", type=int, default=3, help="Default number of documents to retrieve per query.")
    parser.add_argument("--retriever_name", type=str, default="e5", help="Name of the retriever method ('bm25' or a dense model name like 'e5', 'bge').")
    parser.add_argument("--retriever_model", type=str, default="intfloat/e5-base-v2", help="Path or Hugging Face name of the dense retrieval model.")
    parser.add_argument('--faiss_gpu', action='store_true', help='Use GPU for Faiss index if available.')
    parser.add_argument("--pooling", type=str, default="mean", help="Pooling method for dense retriever (e.g., 'mean', 'cls').")
    parser.add_argument("--port", type=int, default=8010, help="Port to run the FastAPI server on.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address to bind the server to.")

    args = parser.parse_args()
    
    # 1. Build the configuration from command-line arguments.
    config = Config(
        retrieval_method=args.retriever_name,
        index_path=args.index_path,
        corpus_path=args.corpus_path,
        retrieval_topk=args.topk,
        faiss_gpu=args.faiss_gpu,
        retrieval_model_path=args.retriever_model,
        retrieval_pooling_method=args.pooling,
        retrieval_query_max_length=256,
        retrieval_use_fp16=True, # Set to True for better performance, requires compatible hardware
        retrieval_batch_size=512,
    )

    # 2. Instantiate the global retriever so it's loaded once and reused across requests.
    print("Initializing retriever...")
    retriever = get_retriever(config)
    print("Retriever initialized successfully.")
    
    # 3. Launch the FastAPI server.
    print(f"Starting server at http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)