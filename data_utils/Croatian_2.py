from .lamol_datasets import LAMOLDataset
import os
import pandas as pd

DATA_DIR = 'datasets/Croatian/Crotian_2_'

BIN_PROMPTS = {
    'crotian2-threats': 'Is this comment threateing?',
    'crotian2-hatespeech': 'Does this comment include hate speech?',
    'crotian2-obscenity': 'Is this comment obscene?',
    'crotian2-deception': 'Does this comment include deceiving or trolling?',
    'crotian2-vulgarity': 'Is this comment vulgar?',
    'crotian2-abuse': 'Is this comment abusive?'
}

BIN_LABEL_MAPPINGS = {
    'crotian2-threats': ["no","yes"],
    'crotian2-hatespeech': ["no","yes"],
    'crotian2-obscenity': ["no","yes"],
    'crotian2-deception': ["no","yes"],
    'crotian2-vulgarity': ["no","yes"],
    'crotian2-abuse': ["no","yes"]
}


class Croatian2Dataset(LAMOLDataset):
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
        label_guide =  {
            'crotian2-threats': 2,
            'crotian2-hatespeech': 3,
            'crotian2-obscenity': 4,
            'crotian2-deception': 5,
            'crotian2-vulgarity': 6,
            'crotian2-abuse': 8
        }
        for _, item in df.iterrows():
            context = item['content']
            answer = item['infringed_on_rule']
            if answer == label_guide[self.task_name]:
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
