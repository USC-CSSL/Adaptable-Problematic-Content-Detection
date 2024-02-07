from .lamol_datasets import LAMOLDataset
import os
import pandas as pd

DATA_DIR = 'datasets/Greek/Greek_1_'

BIN_PROMPT = 'Is it offensive?'

BIN_LABEL_MAPPING = ["no","yes"]


class Greek1Dataset(LAMOLDataset):
    def __init__(self, args, task_name, split, tokenizer, gen_token, full_init=True, use_vocab_space=True, **kwargs):
        super().__init__(args, task_name, split, tokenizer, gen_token, full_init=False, use_vocab_space=use_vocab_space, **kwargs)
        
        # in the original data it is subtask_a
        self.label_column_name = 'offensive' 
        
        if self.split == 'dev':
            self.split = 'val'
        if full_init:
            self.init_data()
        
    def init_data(self):
        data = []
        prompt = BIN_PROMPT
        sep_token = self.tokenizer.sep_token
        
        file_name = os.path.join(DATA_DIR, self.split + '.csv')
        df = pd.read_csv(file_name)
        if self.split == 'train':
            df = self.sample_stratified(df, self.label_column_name, n_samples=1000, random_state=42)
        # TODO: change to following code to apply function to each row
        for _, item in df.iterrows():
            context = item['text']
            answer = item[self.label_column_name]
            if answer == 1:
                answer = 'yes'
            else:
                answer = 'no'
            label_id = BIN_LABEL_MAPPING.index(answer)
            
            entry = {
                'context': context,
                'qas':[{'question': prompt, 'answers': [{'text': answer, 'label': label_id}]}]
            }
            data.append(entry)
        self.data = data
        self.data_tokenization(self.data)
