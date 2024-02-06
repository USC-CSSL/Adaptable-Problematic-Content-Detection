from .lamol_datasets import LAMOLDataset
import os
import pandas as pd

DATA_DIR = 'datasets/Hindi/Hindi_1_'

BIN_PROMPTS = {
    'hindi-fake': 'Does this include fake news?',
    'hindi-hate': 'Does this include hate speech?',
    'hindi-offensive': 'Is this text offensive?',
    'hindi-defamation': 'Does this text include defamation?',
}

BIN_LABEL_MAPPINGS = {
    'hindi-fake': ["no","yes"],
    'hindi-hate': ["no","yes"],
    'hindi-offensive': ["no","yes"],
    'hindi-defamation': ["no","yes"],
}


class Hindi1Dataset(LAMOLDataset):
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
        
        file_name = DATA_DIR + self.split + '.csv'
        df = pd.read_csv(file_name)
        # TODO: change to following code to apply function to each row
        label_guide =  self.task_name.split('-')[1]
        for _, item in df.iterrows():
            context = item['Post']
            answer = item['Labels Set']
            if label_guide in answer:
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