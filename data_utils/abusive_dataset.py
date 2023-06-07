from .lamol_datasets import LAMOLDataset
import os
import pandas as pd

DATA_DIR = 'datasets/abusive/'

BIN_PROMPTS = {
    'abusive-abusive': 'Does this content use hurtful language using profanity that can show a debasement of someone or something, or show intense emotion?',
    'abusive-hateful': 'Does this content use hatred towards a targeted individual or group?',
}

BIN_LABEL_MAPPINGS = {
    'abusive-abusive': ["no", "yes"],
    'abusive-hateful': ["no", "yes"],
}


class AbusiveDataset(LAMOLDataset):
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
            context = item['Tweet']
            answer = item['Label']
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
