from .lamol_datasets import LAMOLDataset
import os
import pandas as pd

DATA_DIR = 'datasets/CAD/'

BIN_PROMPTS = {
    'cad-identitydirectedabuse': "does the text contain identity directed abuse?",
    'cad-persondirectedabuse': "does the text contain have person directed abuse?",
    'cad-affiliationdirectedabuse': "does the text contain have affiliation directed abuse?",
    'cad-counterspeech': "does the text contain have counter speech?",
}

BIN_LABEL_MAPPINGS = {
    'cad-identitydirectedabuse': ["no", "yes"],
    'cad-persondirectedabuse': ["no", "yes"],
    'cad-affiliationdirectedabuse': ["no", "yes"],
    'cad-counterspeech': ["no", "yes"],
}


class CADDataset(LAMOLDataset):
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
        sep_token = self.tokenizer.sep_token
        
        column = self.task_name.split("-")[1]

        file_name = os.path.join(DATA_DIR, self.split + '.csv')
        df = pd.read_csv(file_name)
        # TODO: change to following code to apply function to each row
        for _, item in df.iterrows():
            context = item['text']
            label_id = int(item[column])
            answer = BIN_LABEL_MAPPINGS[self.task_name][int(label_id)]

            if prompt:
                entry = {
                    'context': context,
                    'qas': [{'question': prompt, 'answers': [{'text': answer, 'label': label_id}]}]
                }
            else:
                entry = {
                    'context': '',
                    'qas': [{'question': context, 'answers': [{'text': answer, 'label': label_id}]}]
                }
            data.append(entry)
        self.data = data
        self.data_tokenization(self.data)
