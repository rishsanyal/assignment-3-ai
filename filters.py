from nltk import FreqDist, ConditionalFreqDist
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from math import log
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

import nltk
words = set(nltk.corpus.words.words())

eng_words = stopwords.words("english")

ALLOWED_PART_OF_SPEECH = set([
    'JJ', 'JJR', 'JJS',
    'NN', 'NNS', 'NNP', 'NNPS',
    'RB', 'RBR', 'RBS',
    # 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'
])

def alphabetic(token) :
    try :
        return token.isalpha()
    except:
        return False

def stopword(token) :
    return token not in eng_words

def is_empty(token) :
    return token != ''

def is_small(token) :
    return len(token) > 2

def is_grammer(token):
    return token not in set([
        '.', ',', '!', '?', ':', ';', '(', ')', '[', ']', '{', '}',
        '\'', '\"', '-', '--', '``', '\'\'', '...', '\'s', '\'t',
        'i', 'A', 'a', 'The', 'the'
    ])

def is_allowed_part_of_speech(token):
    return pos_tag([token])[0][1] in ALLOWED_PART_OF_SPEECH


filters = [alphabetic,stopword, is_empty, is_small, is_grammer, is_allowed_part_of_speech]


def trim(token) :
    try :
        return token.strip()
    except :
        return token

def lowercase(token) :
    try :
        return token.lower()
    except :
        return token

def select_features(filters, list_of_tokens) :
    features = []
    for token in list_of_tokens :
        if all([filter(token) for filter in filters]) :
            features.append(token)
    return features

def apply_lemmatizer(token):
    try:
        return WordNetLemmatizer().lemmatize(token)
    except:
        return token

def apply_stemmer(token):
    try:
        return PorterStemmer().stem(token)
    except:
        return token

def remove_grammar(token):
    grammarList = [
        '.', ',', '!', '?', ':', ';', '(', ')', '[', ']', '{', '}',
        '\'', '\"', '-', '--', '``', '\'\'', '...', '\'s', '\'t',
        '\'m', '\'re', '\'ve', '\'d', '\'ll', '\'S', '\'T', '\'M',
        '\'RE', '\'VE', '\'D', '\'LL', 'n\'t', 'N\'T', 'I', 'i', 'A',
        'a', 'The', 'the'
    ]

    for grammar in grammarList:
        token.replace(grammar, '')

    return token

def remove_non_english_words(token):
    return token if token in words else ''

def tokenize(token):
    return word_tokenize(token)[0]

def apply_transforms(transforms, list_of_tokens) :
    changed = []
    for token in list_of_tokens :
        new_token = token
        for transform in transforms :
            new_token = transform(token)
        changed.append(new_token)
    return changed

transforms = [trim, lowercase, apply_stemmer, apply_lemmatizer]










