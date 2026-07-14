
<div align="center"> 
<img src="assets/image.png" width="120px">
</div>
<h1 align="center">🔍📚 Informatica: Open and Scalable Foundations for Deep Research Systems</h1>

<div align="center">
<a href="https://huggingface.co/collections/TommyChien/informatica-68ccc0b35674571aa007d961"><img alt="Models" src="https://img.shields.io/badge/🤗_Models-Infomatica-blue"></a>
<a href="https://huggingface.co/datasets/Lk123/InfoSeek"><img alt="Datasets" src="https://img.shields.io/badge/🤗_Datasets-InfoSeek-yellow"></a>
<a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/License-Apache--2.0-green"></a>
<img alt="Made with Python" src="https://img.shields.io/badge/Made_with-Python-blue">
</div>
    
<h4 align="center">

<p>
<a href="#-overview">Overview</a> |
<a href="#-news">News</a> |
<a href="#-roadmap">Roadmap</a> |
<a href="#-demo">Demo</a> |
<a href="#-misc">Misc</a> |
<a href="#-citation">Citation</a>
</p>
</h4>

## 🔆 Overview
**Informatica** is a comprehensive collection of systematic research projects focused on deep research systems. Our mission is to provide open-source, scalable frameworks, datasets, data synthesis methods, models, and demonstrations for the deep research community.

We are committed to advancing the field of deep research through multi-dimensional investigations, including:

- **Scalable Data Synthesis**: Advanced frameworks for generating high-quality, complexity-controllable research datasets
- **Deep Research Models**: State-of-the-art models trained on structured research tasks
- **Open Datasets**: Publicly available datasets designed for training and evaluating deep research capabilities
- **Research Tools**: Complete toolchains for constructing, training, and deploying deep research systems
- **Interactive Demonstrations**: User-friendly demos showcasing the capabilities of our research systems

Our team continuously explores various aspects of deep research problems, from fundamental question decomposition and reasoning to practical applications in knowledge discovery and information synthesis. Through Informatica, we aim to democratize access to deep research technologies and foster innovation in the broader research community.


## 📰 News
[2026/07] 🧪 We have released [InfoSeek Eval](https://huggingface.co/datasets/Lk123/InfoSeek), a 300-example training-isolated test split for deep search evaluation.

[2026/07] 🎉 Our paper [InfoFlow: Reinforcing Agentic Search Via Reward Density Optimization](https://aclanthology.org/2026.acl-long.467/) has been published at **ACL 2026**.

[2026/04] 🔬 We released [AutoResearchBench](https://github.com/CherYou/AutoResearchBench), a challenging benchmark for scientific deep research and literature discovery. For many frontier models, it can be a tougher test than BrowseComp—ideal for stress-testing agentic research systems. [Code](https://github.com/CherYou/AutoResearchBench) · [Dataset](https://huggingface.co/datasets/Lk123/AutoResearchBench) · [Paper](https://arxiv.org/abs/2604.25256) 🧪

[2026/01] 🎉 Our paper [Open Data Synthesis For Deep Research](https://arxiv.org/abs/2509.00375), introducing InfoSeek, has been accepted to **ICLR 2026**!

[2025/09/19]🎉 Our paper [InForage](https://arxiv.org/abs/2505.09316) has been accepted by NeurIPS 2025 as a **Spotlight** paper! Codes is available [here](https://github.com/VectorSpaceLab/Infomatica/tree/main/research/InForage).


[2025/09/17]🔥 We have released a large-scale dataset for deep research tasks, named [InfoSeek](https://huggingface.co/datasets/Lk123/InfoSeek).

[2025/05/14]🔥 We have released our initial research on agentic search, named [InForage](https://arxiv.org/abs/2505.09316).

## 🗺️ Roadmap

### Initial Research
- [x] Technical Report: InForage - Agentic Search Framework
- [x] NeurIPS 2025 Spotlight Paper Acceptance

### Open and Scalable Data Synthesis
- [x] Open Dataset: InfoSeek
- [x] Data Construction Pipeline
- [x] Scalable Synthesis Framework
- [x] Quality Control Mechanisms

### Model Development
- [x] SFT Training Code
- [x] RL Training Code
- [x] InfoSeeker Model Release
- [ ] Model Evaluation Framework

### Applications
- [ ] Knowledge Discovery Tools
- [ ] Information Synthesis Systems
- [ ] Research Assistant Applications

### Demo and Deployment
- [ ] Interactive Demo Platform
- [ ] API Integration
- [ ] User Interface Development

## 🎯 Demo
We are building a demo page to showcase different agentic search methods and allow hands-on exploration of their capabilities. Each demo will be integrated into a standardized retrieval and web browser interface with comparable settings, enabling comprehensive and fair comparisons across various approaches. This systematic evaluation will help identify strengths and limitations of different methods and advance the state-of-the-art in agentic search. 
## 🌟 Misc

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=VectorSpaceLab/Infomatica&type=Date)](https://www.star-history.com/#VectorSpaceLab/Infomatica&Date)

</div>

## 📄 Citation

AutoResearchBench: [Lei Xiong et al., 2026](https://arxiv.org/abs/2604.25256).

InfoSeek:
```
@misc{xia2025opendatasynthesisdeep,
      title={Open Data Synthesis For Deep Research}, 
      author={Ziyi Xia and Kun Luo and Hongjin Qian and Zheng Liu},
      year={2025},
      url={https://arxiv.org/abs/2509.00375}, 
}
```

InForage:
```
@misc{qian2025scentknowledgeoptimizingsearchenhanced,
      title={Scent of Knowledge: Optimizing Search-Enhanced Reasoning with Information Foraging}, 
      author={Hongjin Qian and Zheng Liu},
      year={2025},
      url={https://arxiv.org/abs/2505.09316}, 
}
```

InfoFlow (ACL 2026):
```
@inproceedings{luo-etal-2026-reinforcing,
    title = "Reinforcing Agentic Search Via Reward Density Optimization",
    author = "Luo, Kun  and
      Qian, Hongjin  and
      Liu, Zheng  and
      Xia, Ziyi  and
      Xiao, Shitao  and
      Cao, Zhao  and
      Bao, Siqi  and
      Zhao, Jun  and
      Liu, Kang",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.467/",
    doi = "10.18653/v1/2026.acl-long.467",
    pages = "10261--10283",
    ISBN = "979-8-89176-390-6"
}
```
