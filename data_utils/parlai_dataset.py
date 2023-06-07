from .lamol_datasets import LAMOLDataset
import os
import pandas as pd

DATA_DIR = 'datasets/'

CHOICE_PROMPTS = {
    'single': 'is this sentence safe or toxic? ',
    'single_adversarial': 'is this sentence safe or toxic? ',
    'multi': 'is this conversation safe or toxic? ',
    'BAD2': 'is this conversation safe or toxic? ',
    'BAD4': 'is this conversation safe or toxic? ',
}

CHOICE_LABEL_MAPPINGS = {
    'single': ['safe','toxic'],
    'single_adversarial': ['safe','toxic'],
    'multi': ['safe','toxic'],
    'BAD2': ['safe','toxic'],
    'BAD4': ['safe','toxic'],
}

BIN_PROMPTS = {
    'single': 'is this text toxic? ',
    'single_adversarial': 'is this text toxic? ',
    'multi': 'is this conversation toxic? ',
    'BAD2': 'is this conversation toxic? ',
    'BAD4': 'is this conversation toxic? ',
}

BIN_LABEL_MAPPINGS = {
    'single': ['safe','toxic'],
    'single_adversarial': ['safe','toxic'],
    'multi': ['safe','toxic'],
    'BAD2': ['safe','toxic'],
    'BAD4': ['safe','toxic'],
}



class ParlaiDataset(LAMOLDataset):
    def __init__(self, args, task_name, split, tokenizer, gen_token, full_init=True, use_vocab_space=True, **kwargs):
        super().__init__(args, task_name, split, tokenizer, gen_token, full_init=False, use_vocab_space=use_vocab_space, **kwargs)
        if self.split == 'dev':
            self.split = 'valid'
        if full_init:
            self.init_data()

    def init_data(self):
        #base_dir = os.path.join(DATA_DIR, 'kilt_{}_wctx'.format(self.task_name), self.task_name)
        #query_file, context_file, answer_file = [os.path.join(base_dir,'{}-{}-{}.tsv'.format(self.task_name, self.split, x))
        #                                         for x in ['query','ctx','answer']]
        #query_lines, context_lines, answer_lines = open(query_file).readlines(), open(context_file).readlines(), \
        #                                           open(answer_file).readlines()
        data = []
        prompt = CHOICE_PROMPTS[self.task_name]
        sep_token = self.tokenizer.sep_token
        
        file_name = os.path.join(DATA_DIR, self.task_name, self.split + '.csv')
        df = pd.read_csv(file_name)
        # TODO: change to following code to apply function to each row
        for _, item in df.iterrows():
            context = item['text']
            answer = item['labels']
            label_id = CHOICE_LABEL_MAPPINGS[self.task_name].index(answer)
            
            if prompt:
                entry = {
                    'context': context,
                    'qas':[{'question': prompt, 'answers': [{'text': answer, 'label': label_id}]}]
                }
            else:
                entry = {
                    'context': '',
                    'qas':[{'question': context, 'answers': [{'text': answer, 'label': label_id}]}]
                }
            data.append(entry)
        self.data = data
        self.data_tokenization(self.data)
