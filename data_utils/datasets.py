from .kilt_datasets import *
from .lamol_datasets import *
from .glue_datasets import *
from .parlai_dataset import ParlaiDataset
from .ghc_dataset import GHCDataset
from .ucc_dataset import UCCDataset
from .us_election_dataset import USElectionDataset
from .personal_attack_dataset import PersonalAttackDataset
from .cmsb_dataset import CSMBDataset
from .stormfront_dataset import StormfrontDataset
from .cad_dataset import CADDataset
from .dygen_dataset import DygenDataset
from .misogyny_dataset import MisogynyDataset
from .hatecheck_dataset import HatecheckDataset
from .conan_dataset import ConanDataset
from .harassment_dataset import HarassmentDataset
from .abusive_dataset import AbusiveDataset
from .hate_dataset import HateDataset
from .jigsaw_dataset import JigsawDataset
from .leopard_datasets import *
from .mbpa_datasets import *
from .crossfit_qa_datasets import *
from .crossfit_cls_datasets import *
from .Italian_1 import Italian1Dataset
from .Ukranian_1 import UkranianDataset
from .Chinese_1 import Chinese1Dataset
from .Latvian_1 import Latvian1Dataset
from .Turkish_1 import Turkish1Dataset
from .Portuguese_1 import PortugueseDataset 
from .Greek_1 import Greek1Dataset  
from .Danish_1 import Danish1Dataset
from .Albanian_1 import Albanian1Dataset
from .German_1 import German1Dataset
from .Russian_1 import Russian1Dataset
from .Arabic_1 import Arabic1Dataset
from .Estonian_1 import Estonian1Dataset
from .Hindi_1 import Hindi1Dataset
import random

KILT_TASKS = ['fever', 'trex', 'structured_zeroshot',
              'hotpotqa', 'nq', 'eli5', 'wow', 'aidayago2', 'triviaqa']
LAMOL_TASKS = ['sst', 'woz.en', 'srl']
GLUE_TASKS = ['cola', 'sst2', 'mrpc', 'qqp',
              'stsb', 'mnli', 'qnli', 'wnli', 'rte']
GHC_TASKS = ['ghc-hd', 'ghc-cv', 'ghc-vo']
CMSB_TASKS = ['cmsb-sexist']
STORMFRONT_TASKS = ['stormfront']
HARASSMENT_TASKS = ['harassment']
MISOGYNY_TASKS = ['misogyny']
HATE_TASKS = ['hate-offensive', 'hate-hateful']
ABUSIVE_TASKS = ['abusive-abusive', 'abusive-hateful']
US_ELECTION_TASKS = ['us_election-trump',
                     'us_election-biden', 'us_election-west',  'us_election-hof']
CAD_TASKS = ['cad-identitydirectedabuse', 'cad-persondirectedabuse',
             'cad-affiliationdirectedabuse', 'cad-counterspeech']
PERSONAL_ATTACK_TASKS = ['personal_attack-a', 'personal_attack-qa',
                         'personal_attack-ra', 'personal_attack-tpa']
PARLAI_TASKS = ['single', 'single_adversarial', 'multi', 'BAD2', 'BAD4']
UCC_TASKS = ['ucc-antagonize', 'ucc-condescending', 'ucc-dismissive', 'ucc-generalisation',
             'ucc-generalisation_unfair', 'ucc-healthy', 'ucc-hostile', 'ucc-sarcastic']
DYGEN_TASKS = ['dygen-hate',
               'dygen-african',
               'dygen-arab',
               'dygen-asi',
               'dygen-asi.chin',
               'dygen-asi.east',
               'dygen-asi.man',
               'dygen-asi.pak',
               'dygen-asi.south',
               'dygen-asi.wom',
               'dygen-asylum',
               'dygen-bis',
               'dygen-bla',
               'dygen-bla.man',
               'dygen-bla.wom',
               'dygen-dis',
               'dygen-eastern.europe',
               'dygen-ethnic.minority',
               'dygen-for',
               'dygen-gay',
               'dygen-gay.man',
               'dygen-gay.wom',
               'dygen-gendermin',
               'dygen-hispanic',
               'dygen-hitler',
               'dygen-immig',
               'dygen-indig',
               'dygen-indig.wom',
               'dygen-jew',
               'dygen-lgbtq',
               'dygen-mixed.race',
               'dygen-mus',
               'dygen-mus.wom',
               'dygen-nazis',
               'dygen-non.white',
               'dygen-old.people',
               'dygen-pol',
               'dygen-ref',
               'dygen-russian',
               'dygen-trans',
               'dygen-trav',
               'dygen-wom',
               'dygen-animosity',
               'dygen-dehumanization',
               'dygen-derogation',
               'dygen-support',
               'dygen-threatening']
HATECHECK_TASKS = ['hatecheck-hate', 'hatecheck-muslims', 'hatecheck-black',
                   'hatecheck-disabled', 'hatecheck-gay', 'hatecheck-immigrants',
                   'hatecheck-trans', 'hatecheck-women']
CONAN_TASKS = ['conan-jews', 'conan-muslims', 'conan-lgbt',
               'conan-disabled', 'conan-poc', 'conan-migrant', 'conan-woman']
JIGSAW_TASKS = ['jigsaw-toxicity', 'jigsaw-severe_toxicity', 'jigsaw-obscene', 'jigsaw-identity_attack',
                'jigsaw-insult', 'jigsaw-threat', 'jigsaw-asian', 'jigsaw-atheist', 'jigsaw-bisexual',
                'jigsaw-black', 'jigsaw-buddhist', 'jigsaw-christian', 'jigsaw-female', 'jigsaw-heterosexual',
                'jigsaw-hindu', 'jigsaw-homosexual_gay_or_lesbian', 'jigsaw-intellectual_or_learning_disability',
                'jigsaw-jewish', 'jigsaw-latino', 'jigsaw-male', 'jigsaw-muslim', 'jigsaw-other_disability',
                'jigsaw-other_gender', 'jigsaw-other_race_or_ethnicity', 'jigsaw-other_religion',
                'jigsaw-other_sexual_orientation', 'jigsaw-physical_disability', 'jigsaw-psychiatric_or_mental_illness',
                'jigsaw-transgender', 'jigsaw-white']

CROSSFIT_QA_METRICS = {
    'acronym_identification': 'EM',
    'ade_corpus_v2-classification': 'Classification-F1',
    'ade_corpus_v2-dosage': 'EM',
    'ade_corpus_v2-effect': 'EM',
    'adversarialqa': 'QA-F1',
    'aeslc': 'Rouge-L',
    'ag_news': 'Classification-F1',
    'ai2_arc': 'ACC',
    'amazon_polarity': 'Classification-F1',
    'anli': 'Classification-F1',
    'app_reviews': 'Pearson-Correlation',
    'aqua_rat': 'ACC',
    'art': 'ACC',
    'aslg_pc12': 'EM',
    'biomrc': 'QA-F1',
    'blimp-anaphor_gender_agreement': 'ACC',
    'blimp-anaphor_number_agreement': 'ACC',
    'blimp-determiner_noun_agreement_with_adj_irregular_1': 'ACC',
    'blimp-ellipsis_n_bar_1': 'ACC',
    'blimp-ellipsis_n_bar_2': 'ACC',
    'blimp-existential_there_quantifiers_1': 'ACC',
    'blimp-irregular_past_participle_adjectives': 'ACC',
    'blimp-sentential_negation_npi_licensor_present': 'ACC',
    'blimp-sentential_negation_npi_scope': 'ACC',
    'blimp-wh_questions_object_gap': 'ACC',
    'boolq': 'ACC',
    'break-QDMR': 'EM',
    'break-QDMR-high-level': 'EM',
    'circa': 'Classification-F1',
    'climate_fever': 'Classification-F1',
    'codah': 'Classification-F1',
    'common_gen': 'Rouge-L',
    'commonsense_qa': 'ACC',
    'cos_e': 'Rouge-L',
    'cosmos_qa': 'ACC',
    'crawl_domain': 'EM',
    'crows_pairs': 'ACC',
    'dbpedia_14': 'Classification-F1',
    'definite_pronoun_resolution': 'ACC',
    'discovery': 'Classification-F1',
    'dream': 'ACC',
    'duorc': 'QA-F1',
    'e2e_nlg_cleaned': 'Rouge-L',
    'eli5-askh': 'Rouge-L',
    'eli5-asks': 'Rouge-L',  # dev
    'eli5-eli5': 'Rouge-L',
    'emo': 'Classification-F1',
    'emotion': 'Classification-F1',
    'empathetic_dialogues': 'Rouge-L',
    'ethos-directed_vs_generalized': 'Classification-F1',
    'ethos-disability': 'Classification-F1',
    'ethos-gender': 'Classification-F1',
    'ethos-national_origin': 'Classification-F1',
    'ethos-race': 'Classification-F1',
    'ethos-religion': 'Classification-F1',
    'ethos-sexual_orientation': 'Classification-F1',
    'financial_phrasebank': 'Classification-F1',
    'freebase_qa': 'EM',
    'gigaword': 'Rouge-L',
    'glue-cola': 'Matthew-Correlation',
    'glue-mnli': 'ACC',
    'glue-mrpc': 'ACC',
    'glue-qnli': 'ACC',
    'glue-qqp': 'ACC',
    'glue-rte': 'ACC',
    'glue-sst2': 'ACC',
    'glue-wnli': 'ACC',
    'google_wellformed_query': 'ACC',
    'hate_speech18': 'Classification-F1',
    'hate_speech_offensive': 'Classification-F1',
    'hatexplain': 'Classification-F1',
    'health_fact': 'Classification-F1',
    'hellaswag': 'ACC',
    'hotpot_qa': 'QA-F1',
    'imdb': 'Classification-F1',
    'jeopardy': 'EM',
    'kilt_ay2': 'EM',
    'kilt_fever': 'ACC',
    'kilt_hotpotqa': 'EM',
    'kilt_nq': 'EM',
    'kilt_trex': 'EM',
    'kilt_wow': 'Rouge-L',
    'kilt_zsre': 'EM',
    'lama-conceptnet': 'EM',
    'lama-google_re': 'EM',
    'lama-squad': 'EM',
    'lama-trex': 'EM',
    'liar': 'Classification-F1',
    'limit': 'EM',
    'math_qa': 'ACC',
    'mc_taco': 'ACC',
    'medical_questions_pairs': 'ACC',
    'mocha': 'Pearson-Correlation',
    'multi_news': 'Rouge-L',
    'numer_sense': 'EM',
    'onestop_english': 'Classification-F1',
    'openbookqa': 'ACC',
    'paws': 'Classification-F1',
    'piqa': 'ACC',
    'poem_sentiment': 'Classification-F1',
    'proto_qa': 'EM',  # here
    'qa_srl': 'EM',
    'qasc': 'ACC',
    'quail': 'ACC',
    'quarel': 'ACC',
    'quartz-no_knowledge': 'ACC',
    'quartz-with_knowledge': 'ACC',
    'quoref': 'QA-F1',
    'race-high': 'ACC',
    'race-middle': 'ACC',
    'reddit_tifu-title': 'Rouge-L',
    'reddit_tifu-tldr': 'Rouge-L',
    'ropes': 'QA-F1',
    'rotten_tomatoes': 'Classification-F1',
    'samsum': 'Rouge-L',
    'scicite': 'Classification-F1',
    'sciq': 'ACC',
    'scitail': 'Classification-F1',
    'search_qa': 'EM',
    'sick': 'Classification-F1',
    'sms_spam': 'Classification-F1',
    'social_i_qa': 'ACC',
    'spider': 'EM',
    'squad-with_context': 'QA-F1',
    'squad-no_context': 'EM',
    'superglue-cb': 'ACC',
    'superglue-copa': 'ACC',
    'superglue-multirc': 'EM',
    'superglue-record': 'QA-F1',
    'superglue-rte': 'ACC',
    'superglue-wic': 'ACC',
    'superglue-wsc': 'ACC',
    'swag': 'ACC',
    'tab_fact': 'Classification-F1',
    'trec': 'Classification-F1',
    'trec-finegrained': 'Classification-F1',
    'tweet_eval-emoji': 'Classification-F1',
    'tweet_eval-emotion': 'Classification-F1',
    'tweet_eval-hate': 'Classification-F1',
    'tweet_eval-irony': 'Classification-F1',
    'tweet_eval-offensive': 'Classification-F1',
    'tweet_eval-sentiment': 'Classification-F1',
    'tweet_eval-stance_abortion': 'Classification-F1',
    'tweet_eval-stance_atheism': 'Classification-F1',
    'tweet_eval-stance_climate': 'Classification-F1',
    'tweet_eval-stance_feminist': 'Classification-F1',
    'tweet_eval-stance_hillary': 'Classification-F1',
    'tweet_qa': 'QA-F1',
    'web_questions': 'EM',
    'wiki_auto': 'Classification-F1',
    'wiki_bio': 'Rouge-L',
    'wiki_qa': 'Classification-F1',
    'wiki_split': 'Rouge-L',
    'wikisql': 'EM',
    'wino_grande': 'ACC',
    'wiqa': 'ACC',
    'xsum': 'Rouge-L',
    'yahoo_answers_topics': 'Classification-F1',
    'yelp_polarity': 'Classification-F1',
    'yelp_review_full': 'Pearson-Correlation'
}


FIRST_SET_TASKS = ['cad-identitydirectedabuse', 'cad-persondirectedabuse', 'cad-affiliationdirectedabuse', 'cad-counterspeech', 
                   'personal_attack-a', 'personal_attack-qa', 'personal_attack-ra', 'personal_attack-tpa', 
                   'ucc-antagonize', 'ucc-condescending', 'ucc-dismissive', 'ucc-generalisation', 'ucc-generalisation_unfair', 'ucc-healthy', 'ucc-hostile', 'ucc-sarcastic', 
                   'ghc-hd', 'ghc-cv', 'ghc-vo', 
                   'multi', 
                   'dygen-hate', 
                   'abusive-abusive', 'abusive-hateful', 'hate-offensive', 'hate-hateful']

FIRST_SET_FS_TASKS = ['cmsb-sexist', 'misogyny', 'stormfront', 'us_election-trump', 'us_election-biden', 'us_election-west', 
                      'us_election-hof', 'single', 'single_adversarial', 'BAD2', 'BAD4', 
                      'dygen-african', 'dygen-arab', 'dygen-asi', 'dygen-asi.chin', 'dygen-asi.east', 
                      'dygen-asi.man', 'dygen-asi.pak', 'dygen-asi.south', 'dygen-asi.wom', 
                      'dygen-bis', 'dygen-bla', 'dygen-bla.wom', 
                      'dygen-dis', 'dygen-for', 
                      'dygen-gay', 'dygen-gay.man', 'dygen-gay.wom', 'dygen-gendermin', 
                      'dygen-hispanic', 'dygen-hitler', 'dygen-immig', 'dygen-indig', 
                      'dygen-indig.wom', 'dygen-jew', 'dygen-lgbtq', 'dygen-mixed.race', 
                      'dygen-mus', 'dygen-mus.wom', 'dygen-non.white', 
                      'dygen-old.people', 'dygen-pol', 'dygen-ref', 'dygen-russian', 
                      'dygen-trans', 'dygen-trav', 'dygen-wom', 'dygen-animosity', 
                      'dygen-dehumanization', 'dygen-derogation', 'dygen-support', 
                      'dygen-threatening', 'hatecheck-hate', 'hatecheck-muslims', 
                      'hatecheck-black', 'hatecheck-disabled', 'hatecheck-gay', 'hatecheck-immigrants',
                      'hatecheck-trans', 'hatecheck-women', 'conan-jews', 'conan-muslims', 'conan-lgbt',
                      'conan-disabled', 'conan-poc', 'conan-migrant', 'conan-woman']

SECOND_SET_TASKS = ['jigsaw-obscene', 'ucc-generalisation_unfair', 'hate-hateful', 'dygen-hate',
                    'ucc-healthy', 'ghc-cv', 'cad-counterspeech', 'ucc-condescending', 
                    'personal_attack-qa', 'ucc-hostile', 'jigsaw-severe_toxicity', 'ucc-antagonize',
                    'jigsaw-toxicity', 'personal_attack-tpa', 'cad-affiliationdirectedabuse', 
                    'ucc-generalisation', 'ghc-hd', 'hate-offensive', 'abusive-hateful', 
                    'jigsaw-threat', 'ucc-dismissive', 'personal_attack-a', 'cad-persondirectedabuse',
                    'jigsaw-insult', 'ucc-sarcastic', 'ghc-vo', 'abusive-abusive', 
                    'personal_attack-ra', 'cad-identitydirectedabuse', 'jigsaw-identity_attack']

SECOND_SET_FS_TASKS = FIRST_SET_FS_TASKS + ['multi']

THIRD_SET_TASKS = ['jigsaw-obscene', 'ucc-generalisation_unfair', 'hate-hateful', 'dygen-hate',
                    'ucc-healthy', 'jigsaw-threat', 'ucc-condescending', 
                    'ucc-hostile', 'ucc-antagonize', 'jigsaw-identity_attack',
                    'jigsaw-toxicity', 'personal_attack-tpa', 'cad-affiliationdirectedabuse', 
                    'ucc-generalisation', 'ghc-hd', 'hate-offensive', 'abusive-hateful', 
                    'ucc-dismissive', 'personal_attack-a', 'cad-persondirectedabuse',
                    'jigsaw-insult', 'ucc-sarcastic', 'ghc-vo', 'abusive-abusive', 
                    'personal_attack-ra', 'cad-identitydirectedabuse']

THIRD_SET_FS_TASKS = SECOND_SET_FS_TASKS + ['cad-counterspeech', 'ghc-cv']

FILTERED_THIRD_SET_FS_TASKS = ['cmsb-sexist', 'misogyny', 'stormfront',
                                'us_election-hof', 'single', 'single_adversarial', 'BAD2', 'BAD4', 'multi',
                                'dygen-african', 'dygen-arab', 'dygen-asi', 'dygen-asi.chin', 'dygen-asi.east', 
                                'dygen-asi.south', 'dygen-asylum', 'dygen-bla', 'dygen-bla.man', 'dygen-bla.wom', 
                                'dygen-dis', 'dygen-eastern.europe', 'dygen-ethnic.minority', 'dygen-for', 
                                'dygen-gay', 'dygen-gay.man', 'dygen-gay.wom', 'dygen-gendermin', 'dygen-immig', 'dygen-indig', 
                                'dygen-jew', 'dygen-mixed.race', 'dygen-mus', 'dygen-mus.wom', 'dygen-nazis', 'dygen-non.white', 
                                'dygen-ref', 'dygen-trans', 'dygen-trav', 'dygen-wom', 'dygen-animosity', 
                                'dygen-dehumanization', 'dygen-derogation', 'dygen-support', 'dygen-threatening', 
                                'hatecheck-hate', 'hatecheck-muslims', 
                                'hatecheck-black', 'hatecheck-disabled', 'hatecheck-gay', 'hatecheck-immigrants',
                                'hatecheck-trans', 'hatecheck-women', 'conan-jews', 'conan-muslims', 'conan-lgbt',
                                'conan-disabled', 'conan-poc', 'conan-migrant', 'conan-woman', 'cad-counterspeech', 'ghc-cv']

PILOT_SET_TASKS = [ 'ucc-generalisation_unfair', 'dygen-hate','cad-affiliationdirectedabuse', 'jigsaw-insult']
PILOT_SET_FS_TASKS = ['cmsb-sexist', 'misogyny', 'stormfront', 'conan-poc', 'BAD4', 'us_election-hof']

FORTH_SET = ['cad-identitydirectedabuse', 'personal_attack-ra', 'jigsaw-obscene', 'ucc-healthy', 
             'jigsaw-toxicity', 'ucc-dismissive', 'ucc-generalisation_unfair', 'personal_attack-a', 
             'ucc-hostile', 'ucc-sarcastic', 'ghc-vo', 'cad-affiliationdirectedabuse', 
             'jigsaw-identity_attack', 'jigsaw-threat', 'abusive-abusive', 'hate-hateful', 
             'ucc-antagonize', 'ghc-hd', 'abusive-hateful', 'dygen-hate', 'cad-persondirectedabuse', 
             'ucc-generalisation', 'personal_attack-tpa', 'hate-offensive', 'ucc-condescending', 
             'jigsaw-insult']

FIFTH_SET = ['cad-persondirectedabuse', 'cad-identitydirectedabuse', 'ucc-generalisation', 'jigsaw-obscene', 
             'personal_attack-ra', 'cad-affiliationdirectedabuse', 'ucc-healthy', 'hate-offensive', 
             'abusive-abusive', 'jigsaw-toxicity', 'ghc-hd', 'ucc-condescending', 'ghc-vo', 'dygen-hate', 
             'ucc-generalisation_unfair', 'personal_attack-a', 'personal_attack-tpa', 'ucc-antagonize', 
             'jigsaw-identity_attack', 'ucc-hostile', 'hate-hateful', 'jigsaw-insult', 'abusive-hateful', 
             'jigsaw-threat', 'ucc-dismissive', 'ucc-sarcastic']

PUBLISHED_TEMPORAL_SET = ['personal_attack-a', 'personal_attack-tpa', 'personal_attack-ra', 
                          'jigsaw-threat', 'jigsaw-insult', 'jigsaw-toxicity', 'jigsaw-identity_attack', 'jigsaw-obscene', 
                          'abusive-abusive', 'abusive-hateful', 
                          'ghc-hd', 'ghc-vo', 
                          'ucc-hostile', 'ucc-generalisation_unfair', 'ucc-dismissive', 'ucc-antagonize',  'ucc-condescending', 'ucc-sarcastic', 'ucc-healthy', 'ucc-generalisation', 
                          'dygen-hate', 
                          'cad-persondirectedabuse', 'cad-identitydirectedabuse', 'cad-affiliationdirectedabuse',
                          'hate-offensive',  'hate-hateful'
                          ]

ITALIAN_TASKS = ['italian']
UKRANINAN_TASKS = ['ukraninan']
CHINESE_TASKS = ['chinese']
LATVIAN_TASKS = ['latvian']
TURKISH_TASKS = ['turkish']
PORTUGUESE_TASKS = ['potuguese-homophobia', 'potuguese-obscene', 'potuguese-insult', 'potuguese-racism', 'potuguese-misogyny', 'potuguese-xenophobia', 'potuguse-hate']
GREEK_TASKS = ['greek']
DANISH_TASKS = ['danish']
ALBANIAN_TASKS = ['albanian']
GERMAN_TASKS = ['german-sexism', 'german-racism', 'german-threat', 'german-insult', 'german-profanity', 'german-offensive']
RUSSIAN_TASKS = ['russian']
ARABIC_TASKS = ['arabic']
ESTONIAN_TASKS = ['estonian']
HINDI_TASKS = ['hindi-fake', 'hindi-hate', 'hindi-offensive', 'hindi-defamation']

MULTILINGUAL_HAMILTONIAN = ['russian', 'arabic', 'estonian', 'chinese', 'stormfront', 'albanian', 
                            'italian', 'danish', 'ukraninan', 'latvian', 'german', 'portuguese', 
                            'greek', 'hindi', 'turkish']

MULTILINGUAL_RANDOM_1 = ['arabic', 'estonian', 'hindi', 'german', 'albanian', 'russian', 
                         'stormfront', 'greek', 'danish', 'turkish', 'latvian', 
                         'portuguese', 'italian', 'ukraninan', 'chinese']

def task_collection_to_tasks(collection_full_name):
    items = collection_full_name.split(':')
    collection_name = items[0]

    tasks = None
    if collection_name == 'second_set':
        tasks = SECOND_SET_TASKS
    elif collection_name == 'first_set_fs':
        tasks = FIRST_SET_FS_TASKS
    elif collection_name == 'first_set':
        tasks = FIRST_SET_TASKS
    elif collection_name == 'third_set':
        tasks = THIRD_SET_TASKS
    elif collection_name == 'third_set_fs':
        tasks = FILTERED_THIRD_SET_FS_TASKS
    elif collection_name == 'forth_set':
        tasks = FORTH_SET
    elif collection_name == 'fifth_set':
        tasks = FIFTH_SET
    elif collection_name == 'published_temporal_set':
        tasks = PUBLISHED_TEMPORAL_SET
    elif collection_name == 'pilot':
        tasks = PILOT_SET_TASKS
    elif collection_name == 'pilot_fs':
        tasks = PILOT_SET_FS_TASKS
    elif collection_name == 'multilingual_hamiltonian':
        tasks = MULTILINGUAL_HAMILTONIAN
    elif collection_name == 'multilingual_random_1':
        tasks = MULTILINGUAL_RANDOM_1
    if len(items) > 1:
        start = int(items[1])
        stop = int(items[2])
        tasks = tasks[start:stop]

    return tasks


def get_main_metrics(task_name):
    if task_name in CROSSFIT_QA_METRICS:
        met = CROSSFIT_QA_METRICS[task_name]
        if met in ['EM', 'ACC']:
            return 'em'
        else:
            return 'f1'
    return 'f1'


def get_dataset(args, task_name, split, tokenizer: PreTrainedTokenizer,
                gen_token=None, full_init=True, **kwargs):
    # if task_name in KILT_TASKS:
    #     DATASET_CLS = KILTDatasetWOContext
    # elif task_name in LAMOL_TASKS:
    #     DATASET_CLS = LAMOLDataset
    # elif task_name in GLUE_TASKS:
    #     DATASET_CLS = GLUEDataset
    # elif task_name in LEOPARD_TASKS and (not args.task_collection or args.task_collection == 'leopard'):
    #     DATASET_CLS = LeopardDataset
    # elif task_name in MBPA_TASKS:
    #     DATASET_CLS = MBPADataset
    # elif task_name in CROSSFIT_QA_TEST_TASKS or task_name in CROSSFIT_QA_TRAIN_TASKS and \
    #         (not args.task_collection or args.task_collection in ['crossfit_qa_train', 'crossfit_qa_test']):
    #     DATASET_CLS = CrossFitQADataset
    # elif task_name in CROSSFIT_CLS_TEST_TASKS or task_name in CROSSFIT_CLS_TRAIN_TASKS and \
    #         (not args.task_collection or args.task_collection in ['crossfit_cls_train', 'crossfit_cls_test']):
    #     DATASET_CLS = CrossFitCLSDataset
    if task_name in PARLAI_TASKS:
        DATASET_CLS = ParlaiDataset
    elif task_name in GHC_TASKS:
        DATASET_CLS = GHCDataset
    elif task_name in PERSONAL_ATTACK_TASKS:
        DATASET_CLS = PersonalAttackDataset
    elif task_name in UCC_TASKS:
        DATASET_CLS = UCCDataset
    elif task_name in CMSB_TASKS:
        DATASET_CLS = CSMBDataset
    elif task_name in US_ELECTION_TASKS:
        DATASET_CLS = USElectionDataset
    elif task_name in STORMFRONT_TASKS:
        DATASET_CLS = StormfrontDataset
    elif task_name in CAD_TASKS:
        DATASET_CLS = CADDataset
    elif task_name in DYGEN_TASKS:
        DATASET_CLS = DygenDataset
    elif task_name in MISOGYNY_TASKS:
        DATASET_CLS = MisogynyDataset
    elif task_name in HATECHECK_TASKS:
        DATASET_CLS = HatecheckDataset
    elif task_name in CONAN_TASKS:
        DATASET_CLS = ConanDataset
    elif task_name in HARASSMENT_TASKS:
        DATASET_CLS = HarassmentDataset
    elif task_name in ABUSIVE_TASKS:
        DATASET_CLS = AbusiveDataset
    elif task_name in HATE_TASKS:
        DATASET_CLS = HateDataset
    elif task_name in JIGSAW_TASKS:
        DATASET_CLS = JigsawDataset
    elif task_name in ITALIAN_TASKS:
        DATASET_CLS = Italian1Dataset
    elif task_name in UKRANINAN_TASKS:
        DATASET_CLS = UkranianDataset
    elif task_name in CHINESE_TASKS:
        DATASET_CLS = Chinese1Dataset
    elif task_name in LATVIAN_TASKS:
        DATASET_CLS = Latvian1Dataset
    elif task_name in TURKISH_TASKS:
        DATASET_CLS = Turkish1Dataset
    elif task_name in PORTUGUESE_TASKS:
        DATASET_CLS = PortugueseDataset
    elif task_name in GREEK_TASKS:
        DATASET_CLS = Greek1Dataset
    elif task_name in DANISH_TASKS:
        DATASET_CLS = Danish1Dataset
    elif task_name in ALBANIAN_TASKS:
        DATASET_CLS = Albanian1Dataset
    elif task_name in GERMAN_TASKS:
        DATASET_CLS = German1Dataset
    elif task_name in RUSSIAN_TASKS:
        DATASET_CLS = Russian1Dataset
    elif task_name in ARABIC_TASKS:
        DATASET_CLS = Arabic1Dataset
    elif task_name in ESTONIAN_TASKS:
        DATASET_CLS = Estonian1Dataset
    elif task_name in HINDI_TASKS:
        DATASET_CLS = Hindi1Dataset
    else:
        raise NotImplementedError
    dataset = DATASET_CLS(args, task_name, split, tokenizer,
                          gen_token, full_init=full_init, **kwargs)
    return dataset
