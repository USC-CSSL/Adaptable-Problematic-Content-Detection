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

3.  Once you have all datasets, you can use our notebook in [`PROJECT_DIR/notebooks/data_preparation.ipynb`](https://github.com/Ali-Omrani/Continual-Problematic-Content-Detection-Benchmark/blob/main/notebooks/data_preparation.ipynb) to prepare all datasets used in the benchmark.

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

## Running your own experiments

We describe the logic of each script to streamline running your own experiments. Let's assume we want to use the upstream task sequence described in the paper (figure below)

![alt text](https://github.com/Ali-Omrani/Continual-Problematic-Content-Detection-Benchmark/blob/main/Figure%202.jpeg)


There are two parameters in the config that can be used to create the task squence, **tasks** or **task_collection**.
To use the tasks parameter, you need to pass the squence of the task in order by space between them as follow:

```
python run_model.py --tasks jigsaw-obscene ucc-generalisation_unfair hate-hateful ...
```

We created our upstream task as a _third_set_ task collection, therefore, you can pass the task_collection parameter as follow:

```
python run_model.py --task_collection third_set ....
```

This design allows you to easily explore various experimental setups with our benchmark. 

### Adding new Task Collection

To create your own tasks squence, you can add another task*collection in the \_data_utils/dataset.py* file and change _task_collection_to_tasks_ function on the same file to make the code run with the new task colleciton.

### Adding New Datasets and Tasks

To add new datset you need to write a dataset adapter, there are three steps that you need to follow:

1. Choose a unique name id for the new dataset and create a python file with the same _NameID_dataset.py_ in data_utils directory.
2. Create a class for the new dataset which inherit the LAMOLDataset, write the init*data file which creates a list of entries (like below) based on task name and pass them to \_self.data_tokenization* function.
3. Add your dataset tasks on top of _data_utils/datasets.py_ file, and your new dataset class in _get_dataset_ function.

```
entry = {'context': Text,
         'qas':[{'question': "", 'answers': [{'text': answer, 'label': label_id}]}]
        }
```

## Adding New Models and Algorithms

## Contributions

We are seeking to continually enhance the Continual Learning Benchmark for Problematic Content Detection (CLB-PCD) with additional tasks, models, and algorithms! If you would like to introduce a new task, model, or algorithm to the CLB-PCD benchmark, please create a Pull Request from your CLB-PCD fork. We will make an effort to incorporate your contribution and enable other researchers to explore it.

If you have any questions please contact Ali Omrani ([aromani@usc.edu](mailto:aomrani@usc.edu)) and Alireza S. Ziabari ([ziabari@usc.edu](mailto:ziabari@usc.edu)).
