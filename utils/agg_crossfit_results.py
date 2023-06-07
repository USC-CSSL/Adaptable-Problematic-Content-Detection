import pandas as pd
import importlib
import os, re
# from metrics.squad_f1 import compute_f1, f1_score_csv_simple
from sklearn.metrics import f1_score, roc_auc_score
import metrics.em
importlib.reload(metrics.em)
from metrics.em import majority_acc
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from itertools import product
import IPython

def match_files(base_dir, with_step=False, is_few_shot=False, load_task=False, postfix='', with_dash=False):
    files = os.listdir(base_dir)
    # print(files)
    pattern = 'results_task_(\d+)_(\w+)_task_([-\d]+).csv'
    value_keys = ['task_id', 'task_name', 'at_task']
    if not load_task:
        if with_step:
            if with_dash:
                pattern = 'results_task_(\d+)_(\w+)-(\w+)_task_(\d+)_step_([\d]+){}.csv'.format(postfix)
                print(pattern)
                value_keys = ['task_id', 'task_name', 'task_detail', 'at_task', 'step']
            else:        
                pattern = 'results_task_(\d+)_(\w+)_task_(\d+)_step_([\d]+){}.csv'.format(postfix)
                print(pattern)
                value_keys = ['task_id', 'task_name', 'at_task', 'step']
                
        if is_few_shot:
            if with_dash:
            # pattern = 'results_task_(\d+)_(\w+)_(\d+)_(\d+)_task_(\d+)_step_(\d+)_fewshot_at_-1{}.csv'.format(postfix)
                pattern =  "results_task_(\d+)_(\w+)-(\w+\.*\w+)_task_(\d+)_step_(\d+)_fewshot_at_-1{}.csv".format(postfix)
            # value_keys = ['task_id', 'task_name', 'split_id', 'shot_num', 'at_task', 'step']
                value_keys = ['task_id', 'task_name', 'task_detail', 'at_task', 'step']
            else:
                pattern = 'results_task_(\d+)_(\w+)_task_(\d+)_step_([\d]+)_fewshot_at_-1{}.csv'.format(postfix)
                print(pattern)
                value_keys = ['task_id', 'task_name', 'at_task', 'step']
    else:
        if not with_step:
            pattern = 'results_task_(\d+)_(\w+)_task_([-\d]+)_load_task_([\d]+){}.csv'.format(postfix)
            value_keys = ['task_id', 'task_name', 'at_task', 'load_task']
        if is_few_shot:
            pattern = 'results_task_(\d+)_(\w+)_(\d+)_(\d+)_task_(\d+)_step_(\d+)_fewshot_at_-1_load_task_([\d]+){}.csv'.format(
                postfix)
            value_keys = ['task_id', 'task_name', 'split_id', 'shot_num', 'at_task', 'step', 'load_task']
        if with_step:
            pattern = 'results_task_(\d+)_(\w+)_task_(\d+)_step_([\d]+)_load_task_([\d]+){}.csv'.format(postfix)
            value_keys = ['task_id', 'task_name', 'at_task', 'step', 'load_task']

    matched = []
    # print(files)
    for file in files:
        obj = re.match(pattern, file)
        if obj is not None:
            # matched.append(obj)
            values = [obj.group(i) for i in range(1, len(value_keys) + 1)]
            item = {k: v for k, v in zip(value_keys, values)}
            item['path'] = os.path.join(base_dir, obj.group(0))
            matched.append(item)

    # print(pattern)
    return matched

def f1_auc_score_csv(file):
    columns = ['text','gt','pred','prob']
    df = pd.read_csv(file, header=None, names=columns)
    f1, auc = f1_score(df['gt'], df['pred']), roc_auc_score(df['gt'], df['prob'])
    maj_acc = majority_acc(df['gt'].tolist())
    return f1, auc, maj_acc

def compute_metrics(path, task=None, metrics_type='auc'):
    f1, auc, maj_acc = f1_auc_score_csv(path)
    if metrics_type == 'f1':
        return f1, maj_acc
    elif metrics_type == 'auc':
        return auc, maj_acc
    # elif metrics_type == 'em':
    #     em_score, maj_score = em_score_csv_simple(path)
    #     return em_score, maj_score
    # elif metrics_type == 'emf1':
    #     em_score, maj_score = em_f1_score_csv_simple(path, task)
    #     return em_score, maj_score
        


def make_few_shot_result_table(base_dir, condition=None, load_task=False, postfix='', metrics_type='auc', with_dash=False):
    matched = match_files(base_dir, is_few_shot=True, load_task=load_task, postfix=postfix, with_dash=with_dash)
    if condition is not None:
        matched = [_ for _ in filter(condition, matched)]
    for item in matched:
        path = item['path']
        try:
            task = item['task_name'] + "-" + item['task_detail'] if "task_detail" in item else item["task_name"]
            # print(task)
            item[metrics_type], item['maj'] = compute_metrics(path=path, task=task, metrics_type=metrics_type)
        except Exception:
            print(path)
            raise
        # item['nf1'] = score

    df = pd.DataFrame.from_records(matched)
    return df


def make_results_table(base_dir, condition=None, load_task=False, metrics_type='auc', with_dash=False):
    matched = match_files(base_dir, load_task=load_task, with_step=True, with_dash=with_dash)
    if len(matched) == 0:
        return pd.DataFrame()
    if condition is not None:
        matched = [_ for _ in filter(condition, matched)]
    for item in matched:
        path = item['path']
        try:
            item[metrics_type], item['maj'] = compute_metrics(path=path, metrics_type=metrics_type)
        except Exception:
            print(path)
            raise
    df = pd.DataFrame.from_records(matched)
    print(df.columns)
    df = df.sort_values(by=['task_id'])
    return df


def make_results_table_instant_performance(base_dir):
    matched = match_files(base_dir, with_step=True)
    matched = [x for x in matched if x['task_id'] == x['at_task']]
    for item in matched:
        path = item['path']
        item['em'], item['maj'] = compute_metrics(path)
    df = pd.DataFrame.from_records(matched)
    df = df.sort_values(by=['task_id'])
    return df


def stat_without_outlier(l):
    q1 = np.quantile(l, 0.25)
    q2 = np.quantile(l, 0.75)
    iqr = q2 - q1
    lb, rb = q1 - 1.5 * iqr, q2 + 1.5 * iqr
    l = [x for x in l if lb <= x <= rb]
    return np.mean(l)


def process_few_shot_df(df_input, step=200, remove_outlier=False):
    # df = df_input.astype({'task_id': int, 'split_id': int, 'shot_num': int, 'at_task': int, 'step': int})
    df = df_input.astype({'task_id': int, "task_detail": str, 'at_task': int, 'step': int})
    
    # df = df.sort_values(by=['task_id','split_id','step'])
    # get performance at certain steps
    # df = df.loc[df['step'] == step]

    # grouped_df = df.groupby(by=['task_name', 'step']).agg(
    grouped_df = df.groupby(by=['task_detail', 'step']).agg(
        {'nf1': stat_without_outlier if remove_outlier else np.mean}).reset_index()

    return grouped_df


def few_shot_performance_by_time(path, load_tasks=[0, 2, 4, 8]):
    scores = []
    for load_task in load_tasks:
        print('process task {}'.format(load_task))
        df_raw = make_few_shot_result_table(path, condition=lambda x: int(x['load_task']) == load_task, load_task=True)
        if load_task == 8 and len(df_raw) == 0:
            df_raw = make_few_shot_result_table(path)
        df = process_few_shot_df(df_raw, step=200)
        scores.append(df)
    return scores


def cl_performance_by_time(path, load_tasks=[0, 1, 2, 3, 4, 5, 6, 7, 8],
                           labels=['HNet-Vanilla', r'HNet$_{\backslash ST}$-Reg', 'HNet-Reg']):
    scores = []
    for load_task in load_tasks:
        print('process task {}'.format(load_task))
        df_raw = make_results_table(path, condition=lambda x: int(x['load_task']) == load_task, load_task=True)
        scores.append(df_raw)
    return scores


def plot_scores(scores_ret_list, x=[0, 2, 4, 8], labels=['HNet-Vanilla', r'HNet$_{\backslash ST}$-Reg', 'HNet-Reg']):
    fig, ax = plt.subplots(figsize=(6, 3))
    all_scores = []
    width = 0.5
    x = np.array(x)
    for i, scores_ret in [_ for _ in enumerate(scores_ret_list)]:
        scores = [x.loc[x['step'] == 400]['nf1'].mean() * 100 for x in scores_ret]
        all_scores.append(scores)
        ax.plot(x + 1, scores, label=labels[i], marker='.')
        # print(np.arange(0,len(x)))
        # ax.bar(x + width * (i - 1), scores, width=width, label=labels[i])
    ax.legend()
    ax.grid()
    ax.set_xticks(x + 1)
    labels = (x + 1).tolist()
    labels[-1] = 'Final'
    ax.set_xticklabels(labels)
    ax.axhline(y=61.30, ls='--', color='gray')
    # ax.set_ylim(55, 73)
    ax.set_xlabel('GLUE Task')
    ax.set_ylabel('Few-Shot Accuracy')


def get_final_epoch_num(df, half=False):
    idx = df.groupby(['task_name'])['step'].transform(max) == df['step'] * (1 if not half else 2)
    return df[idx].reset_index(drop=True)


def average_final_epoch_num_by_task_category(df):
    # df.loc[:,'task_category'] = None
    df.loc[:, 'task_category'] = df['task_name'].map(get_task_category)
    mean_scores = df.groupby(['task_category']).mean()
    cnts = df.groupby(['task_category']).count()['task_name']
    # print(cnts)
    mean_scores.loc[:, 'counts'] = cnts
    mean_scores.loc['mean', 'nf1'] = df['nf1'].mean()
    return mean_scores


def get_task_category(task_name):
    if task_name in ['airline', 'disaster', 'emotion', 'political_audience', 'political_bias', 'political_message',
                     'rating_books',
                     'rating_dvd', 'rating_electronics', 'rating_kitchen']:
        return 'text_classification'
    elif task_name in ['conll', 'restaurant']:
        return 'entity_typing'
    elif task_name in ['scitail']:
        return 'nli'
    elif task_name.startswith('sentiment'):
        return 'sentiment'
    else:
        raise KeyError(task_name)

def summarize_crossfit(path, step=400):
    default_seed = int(path.split('/')[-1])
    all_dfs = []
    all_split_ids = []
    for split_id in range(-1,5):
        try:
            if split_id == -1:
                df = make_few_shot_result_table(path)
            else:
                df = make_few_shot_result_table(path, postfix='_split{}'.format(split_id))
            df = process_few_shot_df(df, remove_outlier=False)
            # ret = get_final_epoch_num(df, half=half)
            ret = df[df['step'] == step]
            print(split_id if split_id!=-1 else default_seed)
            print(ret.mean())
            all_dfs.append(df)
            all_split_ids.append(split_id if split_id !=-1 else default_seed)
        except Exception:
            pass
    return all_dfs, all_split_ids