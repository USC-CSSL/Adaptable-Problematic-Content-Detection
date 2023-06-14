# A Continual Learning Benchmark for Problematic Content Detection

This repository is for a paper **A Continual Learning Benchmark for Problematic Content Detection** submitted to the Datasets & Benchmarks Track at NeurIPS 2023.


![alt text](https://github.com/Ali-Omrani/Continual-Problematic-Content-Detection-Benchmark/blob/main/Figure%201.jpg) 

## Data
To create a local copy of all datasets used in this benchmark follow the instructions below.

1. Download the content of [Google Drive](https://drive.google.com/drive/folders/1SLTprKo6OaDQtpmDXZ5RZu1vrDx0T-LA?usp=sharing), extract the downloaded data and place it under `PROJECT_DIR/datasets`. This includes all datasets that are NOT from Twitter
2. There are two datasets with Twitter as source. These datasets are publicly available as tweet ids which need to be hydrated. 

First get the tweet ids
- [Large-Scale Hate Speech Detection with Cross-Domain Transfer (hate) ](https://aclanthology.org/2022.lrec-1.238.pdf) - Tweet Ids available [here](https://zenodo.org/record/2657374) 
- [Large Scale Crowdsourcing and Characterization of Twitter Abusive Behavior (abusive)](https://arxiv.org/pdf/1802.00393.pdf) Tweet Ids availabe [here](https://github.com/avaapm/hatespeech)

Then you need to hydrate the tweets. You can use any of the various tools available. Here are a few
- [Hydrator](https://github.com/DocNow/hydrator)
- [Twarc](https://github.com/DocNow/twarc) - [Twarc Tutorial](https://scholarslab.github.io/learn-twarc/)


 3. Once you have all datasets, you can use our notebook in [`PROJECT_DIR/notebooks/data_preparation.ipynb`](https://github.com/Ali-Omrani/Continual-Problematic-Content-Detection-Benchmark/blob/main/notebooks/data_preparation.ipynb) to prepare all datasets used in the benchmark.


## Environment
We created our enviroment with conda with Python 3.8.5 and PyTorch 1.7.1.
```bash
conda create --name <env_name> python=3.8.5
conda activate <env_name>
conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 -c pytorch

```
 After creating the environment, used pip to install the packages in the `requirements.txt`.
```bash
pip install -r requirements.txt
```

## Running Experiments

You can use the [provided scripts](https://github.com/Ali-Omrani/Continual-Problematic-Content-Detection-Benchmark/tree/main/script) to replicate all results in our paper.

**Training on upstream**

```
sh ./script/trian.sh
```
```
sh ./script/train_single.sh
```

**Evaluation on fewshots**
```
sh ./script/fs_eval.sh
```
```
sh ./script/fs_single.sh
```

### Running your own experiments
We describe the logic of each script to streamline running your own experiments. Let's assume we want to use the upstream task sequence described in the paper (figure below)

![alt text](https://github.com/Ali-Omrani/Continual-Problematic-Content-Detection-Benchmark/blob/main/Figure%202.jpeg) 

## Adding New Models and Algorithms

## Adding New Datasets and Tasks

## Contributions 
We are seeking to continually enhance the Continual Learning Benchmark for Problematic Content Detection (CLB-PCD) with additional tasks, models, and algorithms! If you would like to introduce a new task, model, or algorithm to the CLB-PCD benchmark, please create a Pull Request from your CLB-PCD fork. We will make an effort to incorporate your contribution and enable other researchers to explore it.
