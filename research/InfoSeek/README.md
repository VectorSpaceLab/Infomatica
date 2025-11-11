<div align="center">
  <img src="./assets/InfoSeek_logo.png" width="150px">
</div>
<h1 align="center">Open Data Synthesis For Deep Research</h1>

<p align="center">
    <a href="https://arxiv.org/abs/2509.00375"><img alt="Build" src="http://img.shields.io/badge/arXiv-InfoSeek-B31B1B.svg?logo=arxiv"></a>
    <a href="https://github.com/VectorSpaceLab/InfoSeek"><img alt="Build" src="https://img.shields.io/badge/Github-InfoSeek-blue?logo=github"></a>
    <a href="https://huggingface.co/datasets/Lk123/InfoSeek"><img alt="Build" src="https://img.shields.io/badge/🤗 Datasets-InfoSeek-yellow"></a>
    <a href="https://opensource.org/license/apache-2-0"><img alt="Build" src="https://img.shields.io/badge/LICENSE-Apache2.0-green.svg">
</p>

## 🔎 Roadmap
**InfoSeek**  is currently under active development, with resources and prototypes continuously being published at this repository.
- [x] Dataset
- [x] Data Construction Codes
- [x] SFT Training Code
- [x] Technical Report
- [x] RL Training Code
- [ ] InfoSeeker Model

## 🔆 Overview
We propose **InfoSeek**, a scalable data synthesis framework for constructing structurally complex Deep Research tasks. InfoSeek designs a dual-agent system to recursively build a *Research Tree* by mining entities and relations from large-scale text, and blurring itermediate vertices to ensure they form valid sub-problems. The agent then transform these trees into natural language questions whose solutions require traversing the entire hierarchy. Using InfoSeek pipeline, we construct a high-quality, complexity-controllable, and intrinsically verifiable dataset.

## 📋 InfoSeek Data

We released [InfoSeek dataset](https://huggingface.co/datasets/Lk123/InfoSeek) on 🤗

### Example 1:
**Question:** What is a species of bird that was named by a person employed under his father between 1818 and 1824, whose wife was a British artist, and which has three subspecies and body length is generally no more than 6 inches?

**Answer:** Russet sparrow

<details>
  <summary>Tree Structure</summary>
  
```
{
  "root": {
    "id": "A",
    "entity": "Russet sparrow",
    "question": "What is a species of bird that was named by a person employed under his father between 1818 and 1824, whose wife was a British artist, and which has three subspecies and body length is generally no more than 6 inches?",
    "claims": [
      { "target_id": "B", "claim": "A was named by B" },
      { "target_id": "C", "claim": "A has three subspecies" },
      { "target_id": "D", "claim": "A's body length is generally no more than 6 inches" }
    ],
    "children": [
      {
        "id": "B",
        "entity": "John Gould",
        "claims": [
          { "target_id": "E", "claim": "B was employed by his father between 1818 and 1824" },
          { "target_id": "F", "claim": "B's wife was F" }
        ],
        "children": [
          { "id": "E", "entity": "None", "claims": [], "children": [] },
          { "id": "F", "entity": "Elizabeth Gould", "claims": [], "children": [] }
        ]
      },
      { "id": "C", "entity": "None", "claims": [], "children": [] },
      { "id": "D", "entity": "None", "claims": [], "children": [] }
    ]
  }
}
```

```
(A: Russet sparrow)
 │
 │
 │── [claim] "was named by" ──> (B: John Gould)
 │    │
 │    │
 │    │── [claim] "was employed by his father (1818-1824)"
 │    │
 │    │
 │    │── [claim] "wife was" ──> (F: Elizabeth Gould)
 │
 │
 │── [claim] "has three subspecies"
 │
 │
 │── [claim] "body length is generally no more than 6 inches"
```
</details>

### Example 2:

**Question:** What is a women's football team whose first goals in the 2. Bundesliga were scored by a player born in Korogocho, who was discovered and developed by the Mathare Youth Sports Association?

**Answer:** SV Werder Bremen (women)

<details>
    <summary>Tree Structure</summary>
  
```
{
  "root": {
    "id": "A",
    "entity": "SV Werder Bremen (women)",
    "question": "What is a women's football team whose first goals in the 2. Bundesliga were scored by a player born in Korogocho, who was discovered and developed by the Mathare Youth Sports Association?",
    "claims": [
      { "target_id": "B", "claim": "A's first goals in the 2. Bundesliga were scored by B" }
    ],
    "children": [
      {
        "id": "B",
        "entity": "Doreen Nabwire",
        "claims": [
          { "target_id": "C", "claim": "B was discovered and developed by C" },
          { "target_id": "D", "claim": "B was born in D" }
        ],
        "children": [
          { "id": "C", "entity": "Mathare Youth Sports Association", "claims": [], "children": [] },
          { "id": "D", "entity": "Korogocho", "claims": [], "children": [] }
        ]
      }
    ]
  }
}
```

```
(A: SV Werder Bremen (women))
 │
 │
 │── [claim] "first goals scored by" ──> (B: Doreen Nabwire)
      │
      │
      │── [claim] "discovered and developed by" ──> (C:Mathare Youth Sports Association)
      │
      │
      │── [claim] "was born in" ──> (D: Korogocho)
```
</details>

## 🔆 RL Training

Our implementation is built upon [Search-R1](https://github.com/PeterGriffinJin/Search-R1).  
We will release an updated version with **new verl support** for **Qwen3** training in the near future.

---

### RL Training Environment

```bash
conda create -n infoseek python=3.9
conda activate infoseek

# Install vLLM (choose a version that suits your environment)
pip install vllm==0.6.3   # Compatible alternatives: 0.5.4, 0.4.2, or 0.3.1

# Install dependencies
pip install -r requirements.txt

# Install FlashAttention 2
pip install flash-attn --no-build-isolation

# Logging
pip install wandb
````

---

### Retriever Environment (Optional)

If you plan to use the retriever, we recommend installing PyTorch via conda to better support `faiss-gpu`.

```bash
conda create -n retriever python=3.10
conda activate retriever

# Install PyTorch + CUDA support
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia

# Install retrieval-related libraries
pip install transformers datasets pyserini

# Install GPU-enabled FAISS for efficient RL rollouts
conda install faiss-gpu=1.8.0 -c pytorch -c nvidia

# API support for retrieval service
pip install uvicorn fastapi
```

---

### Quick Start

We use **1 node with 8×H100 GPUs** for training by default.
If you have **two nodes**, the retriever and refiner services can be deployed on the second node.

#### (1) Download Training Data, Indexes, and Corpus

* [https://huggingface.co/datasets/Lk123/m3_Flat_512](https://huggingface.co/datasets/Lk123/m3_Flat_512)
* [https://huggingface.co/datasets/Lk123/wiki-25-512](https://huggingface.co/datasets/Lk123/wiki-25-512)
* [https://huggingface.co/datasets/Lk123/InfoSeek](https://huggingface.co/datasets/Lk123/InfoSeek)

#### (2) Launch the Retrieval Server (GPUs 4–7)

```bash
conda activate retriever
cd InfoSeek/Retrieve
bash retrieval_launch.sh
```

#### (3) Launch the Refiner Server (GPUs 4–7)

```bash
conda activate infoseek
cd InfoSeek/Refine
bash refiner_launch.sh
```

#### (4) Start RL Training (GPUs 0–3)

```bash
conda activate infoseek
bash train_grpo_format.sh
```


## 📊 Performance
Model trained on InfoSeek and our framework shows strong performances on traditional multi-hop benchmarks:

<img src="./assets/results.png" width="800">

Our 3B model shows competitive results on [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus):

<img src="./assets/browsecomp_plus.png" width="800">

## 📄 License
The code and data accompanying this work are released under the [Apache License, Version 2.0](./LICENSE). This permits use, modification, and distribution for research and commercial purposes, provided that proper attribution is given and the terms of the license are followed.

## ❤️ Citing Us
If you find this repository or our work useful, please consider giving a star ⭐ and or citing our work, which would be greatly appreciated:
```
@misc{xia2025opendatasynthesisdeep,
      title={Open Data Synthesis For Deep Research}, 
      author={Ziyi Xia and Kun Luo and Hongjin Qian and Zheng Liu},
      year={2025},
      eprint={2509.00375},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.00375}, 
}
```
