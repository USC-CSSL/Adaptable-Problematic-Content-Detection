import csv
from sklearn.metrics import f1_score, roc_auc_score
import IPython
import pandas as pd

def get_words(tokens):
    ret = []
    for token in tokens:
        if token == '</s>':
            break
        ret.append(token)
    return ret

def exact_match_acc(preds, gts):
    s, n = 0, 0
    for pred, gt in zip(preds, gts):
        if type(pred) is list:
            pred_words = get_words(pred)
            gt_words = get_words(gt)
            if pred_words == gt_words:
                s += 1
        else:
            if pred == gt:
                s += 1
        n += 1
    return s / n

def em_f1_acc(preds, gts, pos_class=None, neg_class=None):
    s = set(gts)
    # if len(s) != 2 or not any([x.startswith('not') for x in s]):
    #     return exact_match_acc(preds, gts)
    # if pos_class is None:
    #     pos_class = [_ for _ in s if not _.startswith('not')][0]
    #     neg_class = [_ for _ in s if _.startswith('not')][0]
    score = -1
    if len(list(s)) > 2:
        score = f1_score(gts, preds, labels=list(s), average='macro')
    if len(list(s)) == 2:
        if "yes" in list(s):
            pos_class = "yes"
        elif "toxic" in list(s):
            pos_class = "toxic"
        elif "Hateful" in list(s):
            pos_class = "Hateful"
        else:
            pos_class= None
        if pos_class:
            bin_preds = [1 if pred == pos_class else 0 for pred in preds]
            bin_gts = [1 if gt == pos_class else 0 for gt in gts]
            score = f1_score(bin_gts, bin_preds)
    return score

def majority_acc(gts):
    cnts = {}
    for gt in gts:
        if gt not in cnts:
            cnts[gt] = 0
        cnts[gt] += 1
    mx = max([_ for _ in cnts.values()])
    return mx / len(gts)

def em_score_csv_simple(file, with_majority=True):
  with open(file) as f:
    reader = csv.reader(f)
    rows = [_ for _ in reader]
  # cq, gt, pred
  all_scores = []
  gts, preds = [], []
  for cq, gt, pred in rows:
    if gt.startswith('<s>'):
      gt = gt[3:]
      pred = pred[3:]
    gts.append(gt)
    preds.append(pred)
  em_acc = exact_match_acc(preds, gts)
  maj_acc = majority_acc(gts)
  return em_acc, maj_acc

def em_f1_score_csv_simple(file, task=None):
    with open(file) as f:
        reader = csv.reader(f)
        rows = [_ for _ in reader]
    # cq, gt, pred
    all_scores = []
    gts, preds = [], []
    for cq, gt, pred in rows:
        if gt.startswith('<s>'):
            gt = gt[3:]
            pred = pred[3:]
        gts.append(gt)
        preds.append(pred)
    # pos_class, neg_class = get_taks_pos_neg_classes(task)
    # print("we are here! 1 ")
    em_acc = em_f1_acc(preds, gts)
    maj_acc = majority_acc(gts)
    return em_acc, maj_acc

def get_taks_pos_neg_classes(task):
    if task == '':
        return None, None
    return None, None