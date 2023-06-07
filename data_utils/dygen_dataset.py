from .lamol_datasets import LAMOLDataset
import os
import pandas as pd

# (the first round of collection didn't have this info  about targte groups so the data for binary and targets are different)

DATA_DIR_BIN = 'datasets/dygen/dygen_bin/'
DATA_DIR_TAR = 'datasets/dygen/dygen_tar/'

BIN_PROMPTS = {
    'dygen-hate': 'Does this text contain a hateful content?',
    'dygen-african': 'Does this text attack Africans?',
    'dygen-arab': 'Does this text attack Arabs?',
    'dygen-asi': 'Does this text attack Asians?',
    'dygen-asi.chin': 'Does this text attack Chinese people?',
    'dygen-asi.east': 'Does this text attack East Asians?',
    'dygen-asi.man': 'Does this text attack Asian man?',
    'dygen-asi.pak': 'Does this text attack Pakistanis?',
    'dygen-asi.south': 'Does this text attack South Asians?',
    'dygen-asi.wom': 'Does this text attack Asian women?',
    'dygen-asylum': 'Does this text attack Asylum seekers?',
    'dygen-bis': 'Does this text attack Bisexuals?',
    'dygen-bla': 'Does this text attack Black people?',
    'dygen-bla.man': 'Does this text attack Black men?',
    'dygen-bla.wom': 'Does this text attack Black women?',
    'dygen-dis': ' Does this text attack people with disabilities?',
    'dygen-eastern.europe': 'Does this text attack Eastern Europeans?',
    'dygen-ethnic.minority': 'Does this text attack Ethnic Minorities?',
    'dygen-for': 'Does this text attack Foreigners?',
    'dygen-gay': 'Does this text attack gays?',
    'dygen-gay.man': 'Does this text attack gay men?',
    'dygen-gay.wom': 'Does this text attack gay women?',
    'dygen-gendermin': 'Does this text attack Gender minorities?',
    'dygen-hispanic': 'Does this text attack Hispanics?',
    'dygen-hitler': 'Does this text support hitler?',
    'dygen-immig': 'Does this text attack Immigrants?',
    'dygen-indig': 'Does this text attack Indigenous people?',
    'dygen-indig.wom': 'Does this text attack Indigenous Women?',
    'dygen-jew': 'Does this text attack Jewish people?',
    'dygen-lgbtq': 'Does this text attack LGBTQ community?',
    'dygen-mixed.race': 'Does this text attack poeple with mixed race background?',
    'dygen-mus': 'Does this text attack Muslims?',
    'dygen-mus.wom': 'Does this text attack Muslim women?',
    'dygen-nazis': 'Does this text support Nazis?',
    'dygen-non.white': 'Does this text attack Non-whites?',
    'dygen-old.people': 'Does this text attack old people?',
    'dygen-pol': 'Does this text attack Polish people?',
    'dygen-ref': 'Does this text attack Refguees?',
    'dygen-russian': 'Does this text attack Russians?',
    'dygen-trans': "Does this text attack trans people?",
    'dygen-trav': "Does this text attack travellers?",
    'dygen-wom': "Does this text attack women?",
    # The questions for the type of hate came from definitions in paper
    'dygen-animosity': "Does this content express abuse against a group in an implicit or subtle manner?",
    'dygen-dehumanization': "Does this content perceive or treat people as less than human?",
    'dygen-derogation': "Does this content explicitly attack, demonize, demean or insult a group?",
    'dygen-support': "Does this content explicitly glorify, justify or support hateful actions, events, organizations, tropes and individuals?",
    'dygen-threatening': "Does this content express intention to, support for, or encourage inflicting \
    harm on a group, or identified members of the group?"
}

BIN_LABEL_MAPPINGS = {
    'dygen-hate': ["no", "yes"],
    'dygen-african': ["no", "yes"],
    'dygen-arab': ["no", "yes"],
    'dygen-asi': ["no", "yes"],
    'dygen-asi.chin': ["no", "yes"],
    'dygen-asi.east': ["no", "yes"],
    'dygen-asi.man': ["no", "yes"],
    'dygen-asi.pak': ["no", "yes"],
    'dygen-asi.south': ["no", "yes"],
    'dygen-asi.wom': ["no", "yes"],
    'dygen-asylum': ["no", "yes"],
    'dygen-bis': ["no", "yes"],
    'dygen-bla': ["no", "yes"],
    'dygen-bla.man': ["no", "yes"],
    'dygen-bla.wom': ["no", "yes"],
    'dygen-dis': ["no", "yes"],
    'dygen-eastern.europe': ["no", "yes"],
    'dygen-ethnic.minority': ["no", "yes"],
    'dygen-for': ["no", "yes"],
    'dygen-gay': ["no", "yes"],
    'dygen-gay.man': ["no", "yes"],
    'dygen-gay.wom': ["no", "yes"],
    'dygen-gendermin': ["no", "yes"],
    'dygen-hispanic': ["no", "yes"],
    'dygen-hitler': ["no", "yes"],
    'dygen-immig': ["no", "yes"],
    'dygen-indig': ["no", "yes"],
    'dygen-indig.wom': ["no", "yes"],
    'dygen-jew': ["no", "yes"],
    'dygen-lgbtq': ["no", "yes"],
    'dygen-mixed.race': ["no", "yes"],
    'dygen-mus': ["no", "yes"],
    'dygen-mus.wom': ["no", "yes"],
    'dygen-nazis': ["no", "yes"],
    'dygen-non.white': ["no", "yes"],
    'dygen-old.people': ["no", "yes"],
    'dygen-pol':  ["no", "yes"],
    'dygen-ref':  ["no", "yes"],
    'dygen-russian':  ["no", "yes"],
    'dygen-trans': ["no", "yes"],
    'dygen-trav': ["no", "yes"],
    'dygen-wom': ["no", "yes"],
    'dygen-animosity':  ["no", "yes"],
    'dygen-dehumanization':  ["no", "yes"],
    'dygen-derogation': ["no", "yes"],
    'dygen-support': ["no", "yes"],
    'dygen-threatening': ["no", "yes"],
}


# CHOICE_PROMPTS = {
#     'us_election-trump': "Is the tweet text favorable, against, neither, mixed or neutral towards target Trump?",
#     'us_election-biden': "Is the tweet text favorable, against, neither, mixed or neutral towards target Biden?",
#     'us_election-west': "Is the tweet text favorable, against, neither, mixed or neutral towards target West",
#     'us_election-hof': "Is the tweet text hateful and offensive or neither hateful nor offensive."}

# CHOICE_LABEL_MAPPINGS = {
#     'us_election-trump': ['Neither', 'Against', 'Favor', 'Neutral mentions', 'Mixed'],
#     'us_election-biden': ['Neither', 'Against', 'Favor', 'Neutral mentions', 'Mixed'],
#     'us_election-west': ['Neither', 'Favor', 'Neutral mentions', 'Against'],
#     'us_election-hof': ['Non-Hateful', 'Hateful']
# }


class DygenDataset(LAMOLDataset):
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
        if column == 'hate':
            file_name = os.path.join(DATA_DIR_BIN, self.split + '.csv')
        else:
            file_name = os.path.join(DATA_DIR_TAR, self.split + '.csv')
        df = pd.read_csv(file_name)
        # TODO: change to following code to apply function to each row
        for _, item in df.iterrows():
            context = item['text']
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
