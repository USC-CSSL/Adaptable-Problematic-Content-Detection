# Towards a Unified Framework for Adaptable Problematic Content Detection via Continual Learning

**Preprint:** [Towards a Unified Framework for Adaptable Problematic Content Detection via Continual Learning](https://browse.arxiv.org/pdf/2309.16905.pdf) 
by Ali Omrani, Alireza Ziabari, Preni Golazizian, Jeffrey Sorensen, and Morteza Dehghani

Detecting problematic content, such as hate speech, is a multifaceted and ever-changing task, influenced by social dynamics, user populations, diversity of sources, and evolving language. There has been significant efforts, both in academia and in industry, to develop annotated resources that capture various aspects of problematic content. Due to researchers' diverse objectives, the annotations are inconsistent and hence, reports of progress on detection of problematic content are fragmented. This pattern is expected to persist unless we consolidate resources considering the dynamic nature of the problem. We propose integrating the available resources, and leveraging their dynamic nature to break this pattern. In this paper, we introduce a continual learning benchmark and framework for problematic content detection comprising over 84 related tasks encompassing 15 annotation schemas from 8 sources. Our framework creates a novel measure of progress: prioritizing the adaptability of classifiers to evolving tasks over excelling in specific tasks. To ensure the continuous relevance of our framework, we designed it so that new tasks can easily be integrated into the benchmark. Our baseline results demonstrate the potential of continual learning in capturing the evolving content and adapting to novel manifestations of problematic content.

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

We created our upstream task as a `third_set` task collection, therefore, you can pass the `task_collection` parameter as follow:

```
python run_model.py --task_collection third_set ....
```

To create your own tasks squence, you can add another `task_collection` in the [`/data_utils/dataset.py`](https://github.com/Ali-Omrani/Continual-Problematic-Content-Detection-Benchmark/tree/main/data_utils) file and change `task_collection_to_tasks` function on the same file to make the code run with the new task colleciton.

This design allows you to easily explore various experimental setups with our benchmark.

## Adding New Datasets and Tasks

You can easily add a dataset to the benchmark by following the three steps described below:

1. Choose a unique `<Name>` for the new dataset and create a python file with the same `<Name>_dataset.py` in [`data_utils`](https://github.com/Ali-Omrani/Continual-Problematic-Content-Detection-Benchmark/tree/main/data_utils) directory.
2. Create a class for the new dataset which inherit the [`LAMOLDataset`](https://github.com/Ali-Omrani/Continual-Problematic-Content-Detection-Benchmark/blob/main/data_utils/lamol_datasets.py), write the `init_data` function which creates a list of entries (below) based on task name and pass them to `self.data_tokenization` function.

```python
entry = {'context': Text,
         'qas':[{'question': "", 'answers': [{'text': answer, 'label': label_id}]}]
        }
```

3. Add your dataset tasks on top of [`data_utils/datasets.py`](https://github.com/Ali-Omrani/Continual-Problematic-Content-Detection-Benchmark/blob/main/data_utils/datastes.py) file, and your new dataset class in `get_dataset` function.

## Adding New Models and Algorithms

To configure the existing model algorithms, there are several parameters as follow:

- _--no_param_gen_: will not use HNet model (adapter or transformer model)
- _--train_all_: train all the parameter in the model (will not use HNet model)
- _--skip_adapter_: will not pass through adapters.
- _--cl_method_: continual model which can get three values hnet (add regularizer for BiHNet model), ewc (add EWC regularizer to the model), naive (vanilla model).

* _h_l2reg_: regularization coefficient for hnet or ewc methods.

- _--mtl_: create multi task model (need to pass --mtl_task_num)

* _--no_short_term_: will not use short term memory (HNet model instead of BiHNet)
* _--hard_long_term_: compute long term memory
* _--train_task_embs_: learn task embedding with embedding size of _--long_term_task_emb_num_

To change the model architecture, there are the following options:

- _--adapter_dim_: adapter hidden layer size
- _--generator_hdim_: weight generator hidden layer size

To write new model, you need to write a continual model in [`nets/cl_model.py`](https://github.com/Ali-Omrani/Continual-Problematic-Content-Detection-Benchmark/blob/main/nets/cl_model.py), and configure your model training procedure in `train` function in [`run_model.py`](https://github.com/Ali-Omrani/Continual-Problematic-Content-Detection-Benchmark/blob/main/run_model..py).

## Contributions

We are seeking to continually enhance the Continual Learning Benchmark for Problematic Content Detection (CLB-PCD) with additional tasks, models, and algorithms! If you would like to introduce a new task, model, or algorithm to the CLB-PCD benchmark, please create a Pull Request from your CLB-PCD fork. We will make an effort to incorporate your contribution and enable other researchers to explore it.

If you have any questions please contact Ali Omrani ([aromani@usc.edu](mailto:aomrani@usc.edu)) and Alireza S. Ziabari ([salkhord@usc.edu](mailto:salkhord@usc.edu)).

## Citation
If you use the code or findings from this repository in your work, please consider citing our paper:

```plaintext
@article{omrani2023towards,
  title={Towards a Unified Framework for Adaptable Problematic Content Detection via Continual Learning},
  author={Ali Omrani and Alireza S. Ziabari and Preni Golazizian and Jeffrey Sorensen and Morteza Dehghani},
  journal={arXiv preprint arXiv:2309.16905},
  year={2023}
}
