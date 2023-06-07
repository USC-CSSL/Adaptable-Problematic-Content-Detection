from .lamol_datasets import LAMOLDataset
import os
import pandas as pd

# (the first round of collection didn't have this info  about targte groups so the data for binary and targets are different)

DATA_DIR = "datasets/jigsaw/"

BIN_PROMPTS = {
    'jigsaw-toxicity': 'Is this comment toxic?',
    'jigsaw-severe_toxicity': 'Does this comment contain severe toxicity?',
    'jigsaw-obscene':  'Is this comment obscene?',
    'jigsaw-identity_attack':  'Is this comment attacking an identity?',
    'jigsaw-insult':  'Is this comment insulting?',
    'jigsaw-threat':  'Is this comment threatening?',
    'jigsaw-asian':  'Does this comment mention Asians?',
    'jigsaw-atheist': 'Does this comment mention atheists?',
    'jigsaw-bisexual': 'Does this comment mention bisexuals?',
    'jigsaw-black': 'Does this comment mention Black people?',
    'jigsaw-buddhist': 'Does this comment mention Buddhists?',
    'jigsaw-christian': 'Does this comment mention Christians?',
    'jigsaw-female': 'Does this comment mention females?',
    'jigsaw-heterosexual': 'Does this comment mention heterosexuals?',
    'jigsaw-hindu': 'Does this comment mention Hindus?',
    'jigsaw-homosexual_gay_or_lesbian': 'Does this comment mention homosexuals, gays, or lesbians?',
    'jigsaw-intellectual_or_learning_disability': 'Does this comment mention people with intellectual or learning disability?',
    'jigsaw-jewish': 'Does this comment mention Jewish people?',
    'jigsaw-latino': 'Does this comment mention Latinos?',
    'jigsaw-male': 'Does this comment mention males?',
    'jigsaw-muslim': 'Does this comment mention Muslims?',
    'jigsaw-other_disability': 'Does this comment mention Hindus?',
    'jigsaw-other_gender': 'Does this comment mention other genders than male and female?',
    'jigsaw-other_race_or_ethnicity': 'Does this comment mention other races or ethnicities than White, Black, Asian, and Latino?',
    'jigsaw-other_religion': 'Does this comment mention other religions than Christian, Jewish, Muslim, Hindu, Buddhist, and atheist?',
    'jigsaw-other_sexual_orientation': 'Does this comment mention other sexual orientations than heterosexual, homosexual, and bisexual?',
    'jigsaw-physical_disability': 'Does this comment mention poeple with physcial disability?',
    'jigsaw-psychiatric_or_mental_illness': 'Does this comment mention people with psychiatric or mental illness?',
    'jigsaw-transgender': 'Does this comment mention transgenders?',
    'jigsaw-white': 'Does this comment mention White people?'
}

BIN_LABEL_MAPPINGS = {
    'jigsaw-toxicity': ['no', 'yes'],
    'jigsaw-severe_toxicity': ['no', 'yes'],
    'jigsaw-obscene': ['no', 'yes'],
    'jigsaw-identity_attack': ['no', 'yes'],
    'jigsaw-insult': ['no', 'yes'],
    'jigsaw-threat': ['no', 'yes'],
    'jigsaw-asian': ['no', 'yes'],
    'jigsaw-atheist': ['no', 'yes'],
    'jigsaw-bisexual': ['no', 'yes'],
    'jigsaw-black': ['no', 'yes'],
    'jigsaw-buddhist': ['no', 'yes'],
    'jigsaw-christian': ['no', 'yes'],
    'jigsaw-female': ['no', 'yes'],
    'jigsaw-heterosexual': ['no', 'yes'],
    'jigsaw-hindu': ['no', 'yes'],
    'jigsaw-homosexual_gay_or_lesbian': ['no', 'yes'],
    'jigsaw-intellectual_or_learning_disability': ['no', 'yes'],
    'jigsaw-jewish': ['no', 'yes'],
    'jigsaw-latino': ['no', 'yes'],
    'jigsaw-male': ['no', 'yes'],
    'jigsaw-muslim': ['no', 'yes'],
    'jigsaw-other_disability': ['no', 'yes'],
    'jigsaw-other_gender': ['no', 'yes'],
    'jigsaw-other_race_or_ethnicity': ['no', 'yes'],
    'jigsaw-other_religion': ['no', 'yes'],
    'jigsaw-other_sexual_orientation': ['no', 'yes'],
    'jigsaw-physical_disability': ['no', 'yes'],
    'jigsaw-psychiatric_or_mental_illness': ['no', 'yes'],
    'jigsaw-transgender': ['no', 'yes'],
    'jigsaw-white': ['no', 'yes']
}


class JigsawDataset(LAMOLDataset):
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
        prompt = BIN_PROMPTS[self.task_name]
        sep_token = self.tokenizer.sep_token
        column = self.task_name.split("-")[1]
        # print(column)

        if self.split == 'train':
            file_name = os.path.join(DATA_DIR, f'{self.split}_{column}.csv')
        else:
            file_name = os.path.join(DATA_DIR, f'{self.split}_sampled.csv')
        
        
        df = pd.read_csv(file_name)
        df = df.dropna(subset=column)
        # TODO : remove the nans for each task
        # TODO: change to following code to apply function to each row
        for _, item in df.iterrows():
            context = item['comment_text']
            # print(self.task_name, column, df.columns)
            label_id = int(item[column])

            # print(label_id)
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
