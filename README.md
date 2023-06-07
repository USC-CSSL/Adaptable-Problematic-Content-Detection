# A Continual Learning Benchmark for Problematic Content Detection

This repository is for a paper submitted to the Datasets & Benchmarks Track at NeurIPS 2023.

## Data
 Our training data stream and few-shot datasets ar .

Please, donwload public datasets that we used in our benchmark from [Google Drive](https://drive.google.com/drive/folders/1SLTprKo6OaDQtpmDXZ5RZu1vrDx0T-LA?usp=sharing), extract the downloaded data and place it under `PROJECT_DIR/datasets`. There are two datasets in our directory named as [hate](https://aclanthology.org/2022.lrec-1.238.pdf), and [abusive](https://arxiv.org/pdf/1802.00393.pdf) which are not available publicly, therefore, you need to hydrate the tweets or get them from the authors, then you can use our notebook in `PROJECT_DIR/notebooks/data_preparation.ipynb` to create the datasets that we used in our paper.


## Environment
We created our enviroment with conda with Python 3.8.5 and PyTorch 1.7.1. After creating the environment, used pip to install the packages in the `requirements.txt`.

## Running Experiments

All the scripts are available in the script to run the models for upstream and downstream models, you can use the following shell files.

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