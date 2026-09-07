# GraphPoison
Analysing poisoning attacks on graph based RAG Systems

This Repo serves multiple purposes: 
1. Reproduce and analyse the detection of GRAG poisoning attacks such as GragPoison or logicPoison https://github.com/Jord8061/logicPoison
2. Creating a unified framework for detection
3. Investigating deliberate version access for detection strategies
4. Analysing topological signals for detection

## Installation

the repo contains all files but requires a "pip install -r requirements.txt" to install the necessary dependencies

# Datasets: 
the used sets are: 

- https://huggingface.co/datasets/Jord8061/datasets?clone=true
- Wikipedia-Vandalims-Corpus WVC

# Usage:

The config.yaml contains all configurable options. After configuring chosing the pipeline stages and options, the main.py can be run

All results such as graphs, parameter choices, and predicted classes can be found in the results folder. 


