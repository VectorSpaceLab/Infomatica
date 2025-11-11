import asyncio
import json
import os
import random
import time
import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from typing import List, AsyncGenerator, Union, Iterable

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.utils import random_uuid
from vllm.outputs import RequestOutput

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Model and Engine Configuration --- 
MODEL_NAME = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
TENSOR_PARALLEL_SIZE = 1
GPU_MEMORY_UTILIZATION = 0.75
MAX_MODEL_LEN = 1024 * 12

# The number of requests to process concurrently in the vLLM engine.
# This value should be less than or equal to `max_num_seqs`.
MAX_CONCURRENT_REQUESTS = 384

# The maximum number of sequences the vLLM scheduler can handle,
# a hard limit based on hardware and model.
MAX_SEQS_IN_ENGINE = 768
# The maximum number of tokens in a batch, tune this based on your GPU and model length.
MAX_BATCHED_TOKENS = 32768


PROMPT_TEMPLATE = """<|im_start|>user
**TASK:**
Synthesize the key information from the **[Retrieved Documents]** that is relevant to the **[Current Query]**. The synthesis should be guided by conducting deep research to uncover the **[Original Question]**.

**INSTRUCTIONS:**
1.  **Extract & Merge:** Identify all relevant facts and combine them. Eliminate redundancy. You should provide information for deep research, not answer to current query or original question.
2.  **Provide Information, Not an Answer:** Your output should be a self-contained block of information, NOT a direct, short answer to the original question or the current query.
3.  **Handle Insufficient Information:** If the documents do not contain relevant information for the query, state that the provided sources are insufficient and suggest that further investigation may be needed. You can also provide some further investigation direction and query rewrite suggestions.
4.  **Format:** Enclose the entire synthesized output within `<information>` and `</information>` tags. Add no other text. For example, <information> Synthesized information for deep research here </information>.


**CONTEXT:**
- **[Original Question]:** {original_question}
- **[Current Query]:** {query}
- **[Retrieved Documents]:** {documents}

**TASK:**
Synthesize the key information from the **[Retrieved Documents]** that is relevant to the **[Current Query]**. The synthesis should be guided by conducting deep research to uncover the **[Original Question]**.

**INSTRUCTIONS:**
1.  **Extract & Merge:** Identify all relevant facts and combine them. Eliminate redundancy. You should provide information for deep research, not answer to current query or original question.
2.  **Provide Information, Not an Answer:** Your output should be a self-contained block of information, NOT a direct, short answer to the original question or the current query.
3.  **Handle Insufficient Information:** If the documents do not contain relevant information for the query, state that the provided sources are insufficient and suggest that further investigation may be needed. You can also provide some further investigation direction and query rewrite suggestions.
4.  **Format:** Enclose the entire synthesized output within `<information>` and `</information>` tags. Add no other text. For example, <information> Synthesized information for deep research here </information>.

**SYNTHESIZED INFORMATION:**
<|im_end|>
<|im_start|>assistant
"""

# --- Initialize Engine ---
logging.info("Initializing vLLM engine...")
engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    tokenizer=MODEL_NAME,
    tensor_parallel_size=TENSOR_PARALLEL_SIZE,
    gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
    max_model_len=MAX_MODEL_LEN,
    trust_remote_code=True,
    max_num_batched_tokens=MAX_BATCHED_TOKENS,
    max_num_seqs=MAX_SEQS_IN_ENGINE,
    dtype="bfloat16",
    enable_chunked_prefill=True,
    # Enable stats logging for monitoring
    disable_log_stats=False
)

engine = AsyncLLMEngine.from_engine_args(engine_args)
app = FastAPI()

# Use the defined semaphore for concurrency control.
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

class SynthesisRequest(BaseModel):
    original_questions: List[str]
    queries: List[str]
    documents: List[str]


async def get_final_output(results_generator: AsyncGenerator[RequestOutput, None]) -> RequestOutput:
    final_output = None
    async for request_output in results_generator:
        # For `generate`, the final output is typically the last one.
        if request_output.finished:
            final_output = request_output
    return final_output

async def process_single_prompt(prompt: str, sampling_params: SamplingParams) -> str:
    request_id = random_uuid()
    output_text = ""
    try:
        async with semaphore:
            # Use asyncio.wait_for for timeout control.
            generator = engine.generate(prompt, sampling_params, request_id)
            final_output = await asyncio.wait_for(get_final_output(generator), timeout=1200.0)
            
            if final_output and final_output.outputs:
                output_text = final_output.outputs[0].text.strip()
                # Randomly log a small sample of the output for spot-checking.
                if random.random() < 0.01:
                    logging.info(f"Sample Output: {output_text[:100]}...")
            else:
                output_text = "[ERROR: No output from vLLM]"
                logging.warning(f"Request {request_id} produced no output.")

    except asyncio.TimeoutError:
        output_text = "[ERROR: Timeout while generating response]"
        logging.error(f"Request {request_id} timed out.")
        # If a timeout occurs, abort the request in the vLLM engine.
        await engine.abort(request_id)
    except Exception as e:
        output_text = f"[ERROR: {type(e).__name__} - {str(e)}]"
        logging.exception(f"An error occurred while processing request {request_id}")
    
    return output_text

def chunked_iterable(iterable: Iterable, size: int):
    """Yield successive n-sized chunks from an iterable."""
    for i in range(0, len(iterable), size):
        yield iterable[i:i + size]

@app.post("/synthesize")
async def synthesize(request: SynthesisRequest):
    start_time = time.time()
    
    if not (len(request.original_questions) == len(request.queries) == len(request.documents)):
        return JSONResponse(
            status_code=400,
            content={"error": "The number of original_questions, queries, and documents must be equal."}
        )
    
    num_prompts = len(request.documents)
    logging.info(f"Received synthesis request with {num_prompts} documents.")

    prompts = [
        PROMPT_TEMPLATE.format(original_question=oq, query=q, documents=d)
        for oq, q, d in zip(request.original_questions, request.queries, request.documents)
    ]

    sampling_params = SamplingParams(
        temperature=0.5,
        top_p=0.95,
        max_tokens=512,
        stop=["<|im_end|>", "</s>"]
    )

    all_results = []
    # Split a large request into smaller chunks, each no larger than our concurrency limit.
    # This feeds tasks to vLLM smoothly instead of in one large burst.
    chunk_size = MAX_CONCURRENT_REQUESTS
    
    for i, prompt_chunk in enumerate(chunked_iterable(prompts, chunk_size)):
        logging.info(f"Processing chunk {i+1} with {len(prompt_chunk)} prompts...")
        
        tasks = [
            process_single_prompt(prompt, sampling_params)
            for prompt in prompt_chunk
        ]
        
        # Process each chunk concurrently.
        chunk_results = await asyncio.gather(*tasks)
        all_results.extend(chunk_results)
        
        # await asyncio.sleep(0.1)

    total_time = time.time() - start_time
    logging.info(f"Finished processing {num_prompts} documents in {total_time:.2f} seconds.")

    return {"search_results_processed": all_results}

@app.get("/health")
async def health_check():
    # A simple health check. Can be extended to check if the model is loaded successfully.
    return Response(status_code=200, content="OK")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")