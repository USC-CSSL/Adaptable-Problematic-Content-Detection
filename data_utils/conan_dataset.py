from .lamol_datasets import LAMOLDataset
import os
import pandas as pd

# (the first round of collection didn't have this info  about targte groups so the data for binary and targets are different)

DATA_DIR = "datasets/conan/"

BIN_PROMPTS = {
    'conan-jews': 'Does this text contain a hateful content toward jewish people?',
    'conan-muslims': 'Does this text contain a hateful content toward muslims?',
    'conan-lgbt':  'Does this text contain a hateful content toward LGBTQ community?',
    'conan-disabled':  'Does this text contain a hateful content toward disabled people?',
    'conan-poc':  'Does this text contain a hateful content toward people of color?',
    'conan-migrant':  'Does this text contain a hateful content toward immigrants?',
    'conan-woman':  'Does this text contain a hateful content toward women?',

}

BIN_LABEL_MAPPINGS = {
    'conan-jews': ['no', 'yes'],
    'conan-muslims': ['no', 'yes'],
    'conan-lgbt': ['no', 'yes'],
    'conan-disabled': ['no', 'yes'],
    'conan-poc': ['no', 'yes'],
    'conan-migrant': ['no', 'yes'],
    'conan-woman': ['no', 'yes'],
}


class ConanDataset(LAMOLDataset):
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

        file_name = os.path.join(DATA_DIR, self.split + '.csv')

        df = pd.read_csv(file_name)
        # TODO: change to following code to apply function to each row
        for _, item in df.iterrows():
            context = item['HATE_SPEECH']
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
