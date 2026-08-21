---
task_categories:
- question-answering
---

# LogicPoison: Logical Attacks on Graph Retrieval-Augmented Generation

This repository contains the datasets for **LogicPoison**, a logical poisoning framework for Graph-based Retrieval-Augmented Generation (GraphRAG) systems. 

- **Paper:** [LogicPoison: Logical Attacks on Graph Retrieval-Augmented Generation](https://huggingface.co/papers/2604.02954)
- **GitHub Repository:** [Jord8061/logicPoison](https://github.com/Jord8061/logicPoison)

## Overview

LogicPoison targets the topological integrity of knowledge graphs used in GraphRAG. Instead of injecting false content, it perturbs the implicit reasoning topology through type-preserving entity swapping. This maintains surface-level textual plausibility while corrupting the logical connections required for multi-hop inference.

The dataset includes processed and poisoned versions of the following benchmarks:
- **HotpotQA**
- **MuSiQue**
- **2WikiMultiHopQA**

## Method

LogicPoison consists of three stages:
1. **Global Logic Poison:** Identifying high-frequency entities acting as global logic hubs.
2. **Query-Centric Logic Poison:** Identifying bridge entities essential for answering multi-hop questions.
3. **Type-Preserving Entity Swapping:** Performing bijective swapping within the same entity type (e.g., PERSON with PERSON) to break logical connectivity while keeping text natural.

## Usage

As specified in the official repository, you can clone the datasets using:

```bash
git clone https://huggingface.co/datasets/Jord8061/datasets
```

## Responsible Use

This project is released for **research and defensive evaluation purposes only**. LogicPoison demonstrates a realistic vulnerability in GraphRAG systems to support the development of more robust pipelines and stronger defense mechanisms. Please do not use this repository to attack real-world systems.

## Citation

```bibtex
@inproceedings{xiao-etal-2026-logicpoison,
    title = "{L}ogic{P}oison: Logical Attacks on Graph Retrieval-Augmented Generation",
    author = "Xiao, Yilin  and
      Chen, Jin  and
      Zhang, Qinggang  and
      Zhang, Yujing  and
      Zhou, Chuang  and
      Yang, Longhao  and
      Ren, Lingfei  and
      Yang, Xin  and
      Huang, Xiao",
    editor = "Liakata, Maria  and
      Moreira, Viviane P.  and
      Zhang, Jiajun  and
      Jurgens, David",
    booktitle = "Proceedings of the 64th Annual Meeting of the {A}ssociation for {C}omputational {L}inguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2026",
    address = "San Diego, California, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.acl-long.252/",
    pages = "5575--5591",
    ISBN = "979-8-89176-390-6"
}
```