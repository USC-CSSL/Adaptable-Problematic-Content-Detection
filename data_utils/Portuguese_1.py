from .lamol_datasets import LAMOLDataset
import os
import pandas as pd

DATA_DIR = 'datasets/Portuguese/Portuguese_1_'

BIN_PROMPTS = {
    'potuguese-homophobia' : 'Is this post homophobic?',
    'potuguese-obscene' : 'Is this post obscene?',
    'potuguese-insult' : 'Is this post insulting?',
    'potuguese-racism' : 'Does this post include racism?',
    'potuguese-misogyny' : 'Does this include misogyny?',
    'potuguese-xenophobia' : 'Does this include xenophobia?',
    'potuguse-hate' : 'Does this include hate speech?'
}

BIN_LABEL_MAPPINGS = {
    'potuguese-homophobia' : ["no","yes"],
    'potuguese-obscene' : ["no","yes"],
    'potuguese-insult' : ["no","yes"],
    'potuguese-racism' : ["no","yes"],
    'potuguese-misogyny' : ["no","yes"],
    'potuguese-xenophobia' : ["no","yes"],
    'potuguse-hate' : ["no","yes"]
}


class PortugueseDataset(LAMOLDataset):
    def __init__(self, args, task_name, split, tokenizer, gen_token, full_init=True, use_vocab_space=True, **kwargs):
        super().__init__(args, task_name, split, tokenizer, gen_token, full_init=False, use_vocab_space=use_vocab_space, **kwargs)
        if self.split == 'dev':
            self.split = 'val'
        if full_init:
            self.init_data()

    def init_data(self):
        data = []
        prompt = BIN_PROMPTS[self.task_name]
        sep_token = self.tokenizer.sep_token
        column = self.task_name.split('-')[1]
        file_name = DATA_DIR + self.split + '.csv'
        df = pd.read_csv(file_name)
        # TODO: change to following code to apply function to each row
        for _, item in df.iterrows():
            context = item['text']
            answer = item[column]
            if answer > 0:
                answer = 'yes'
            else:
                answer = 'no'
            label_id = BIN_LABEL_MAPPINGS[self.task_name].index(answer)
            
            entry = {
                'context': context,
                'qas':[{'question': prompt, 'answers': [{'text': answer, 'label': label_id}]}]
            }
            data.append(entry)
        self.data = data
        self.data_tokenization(self.data)
