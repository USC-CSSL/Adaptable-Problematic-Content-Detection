from .lamol_datasets import LAMOLDataset
import os
import pandas as pd

DATA_DIR = 'datasets/us_election/'

CHOICE_PROMPTS = {
    'us_election-trump': "Is the tweet text favorable, against, neither, mixed or neutral towards target Trump?",
    'us_election-biden': "Is the tweet text favorable, against, neither, mixed or neutral towards target Biden?",
    'us_election-west': "Is the tweet text favorable, against, neither, mixed or neutral towards target West",
    'us_election-hof': "Is the tweet text hateful and offensive or neither hateful nor offensive."}

CHOICE_LABEL_MAPPINGS = {
    'us_election-trump': ['Neither', 'Against', 'Favor', 'Neutral mentions', 'Mixed'],
    'us_election-biden': ['Neither', 'Against', 'Favor', 'Neutral mentions', 'Mixed'],
    'us_election-west': ['Neither', 'Favor', 'Neutral mentions', 'Against'],
    'us_election-hof': ['Non-Hateful', 'Hateful']
}


class USElectionDataset(LAMOLDataset):
    def __init__(self, args, task_name, split, tokenizer, gen_token, full_init=True, use_vocab_space=True, **kwargs):
        super().__init__(args, task_name, split, tokenizer, gen_token,
                         full_init=False, use_vocab_space=use_vocab_space, **kwargs)
        if full_init:
            self.init_data()

    def init_data(self):
        #base_dir = os.path.join(DATA_DIR, 'kilt_{}_wctx'.format(self.task_name), self.task_name)
        # query_file, context_file, answer_file = [os.path.join(base_dir,'{}-{}-{}.tsv'.format(self.task_name, self.split, x))
        #                                         for x in ['query','ctx','answer']]
        # query_lines, context_lines, answer_lines = open(query_file).readlines(), open(context_file).readlines(), \
        #                                           open(answer_file).readlines()
        data = []
        prompt = CHOICE_PROMPTS[self.task_name]
        sep_token = self.tokenizer.sep_token
        column = self.task_name.split("-")[1]

        file_name = os.path.join(DATA_DIR, self.split + '.csv')
        df = pd.read_csv(file_name)
        # TODO: change to following code to apply function to each row
        for _, item in df.iterrows():
            context = item['text']
            answer = item[column]
            label_id = CHOICE_LABEL_MAPPINGS[self.task_name].index(answer)

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
