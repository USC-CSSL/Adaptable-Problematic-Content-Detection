from .lamol_datasets import LAMOLDataset
import os
import pandas as pd

DATA_DIR = 'datasets/hate/'

BIN_PROMPTS = {
    'hate-offensive': 'Does this content humiliate, taunt, discriminate, or insult?',
    'hate-hateful': 'Does this content contain hate speech?',
}

BIN_LABEL_MAPPINGS = {
    'hate-offensive': ["no", "yes"],
    'hate-hateful': ["no", "yes"],
}


class HateDataset(LAMOLDataset):
    def __init__(self, args, task_name, split, tokenizer, gen_token, full_init=True, use_vocab_space=True, **kwargs):
        super().__init__(args, task_name, split, tokenizer, gen_token,
                         full_init=False, use_vocab_space=use_vocab_space, **kwargs)
        if self.split == 'dev':
            self.split = 'val'
        if full_init:
            self.init_data()

    def init_data(self):
        data = []
        prompt = BIN_PROMPTS[self.task_name]
        label_name = self.task_name.split("-")[1]

        file_name = os.path.join(DATA_DIR, self.split + '.csv')
        df = pd.read_csv(file_name)
        for _, item in df.iterrows():
            context = item['text']
            answer = item['label']
            if answer == label_name:
                answer = 'yes'
            else:
                answer = 'no'

            label_id = BIN_LABEL_MAPPINGS[self.task_name].index(answer)

            entry = {
                'context': context,
                'qas': [{'question': prompt, 'answers': [{'text': answer, 'label': label_id}]}]
            }
            data.append(entry)
        self.data = data
        self.data_tokenization(self.data)
