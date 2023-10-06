## Naive Bayes using NLTK
from nltk import FreqDist, ConditionalFreqDist
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
from math import log
from filters import *
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

from sklearn.model_selection import KFold


## assume the training set is:
## a dictionary mapping category names to lists of files.
## e.g. t = {'pos' : ['pos/cv999_13106.txt', 'pos/cv998_14111.txt',...] ,
##           'neg' : ['neg/cv371_8197.txt', 'neg/cv374_26455.txt'] }
## Construct a ConditionalFrequencyDistribution that maps categories onto distributions.

## This should just work. You're welcome to adapt it if you'd like.
def train(training_set, apply_transformation=True) :

    model = ConditionalFreqDist()
    num_words = 0
    for category in training_set :
        files = training_set[category]
        for name in files :
            if apply_transformation:
                word_list = apply_transforms(transforms,select_features(filters, movie_reviews.words(name)))
            else:
                word_list = apply_transforms([],select_features([], movie_reviews.words(name)))

            for word in word_list :
                model[category][word] += 1
                num_words += 1
    return model, num_words


## classify
## Given a list of tokens and a model, return a dictionary mapping categories in the model
# to their log-likelihood.

def classify(model, list_of_tokens, apply_transformation=True) :
    if apply_transformation:
        filtered_tokens = apply_transforms(transforms, select_features(filters, list_of_tokens))
    else:
        filtered_tokens = apply_transforms([], select_features([], list_of_tokens))
    categories = list(model.keys())
    results = {}

    ## TODO: Check what log likelihood is and how to calculate it
    for category in categories:
        results[category] = 0
        for token in filtered_tokens:
            # results[category] += log(model.get(category, {}).get(token, 0) + 1) - log(model[category].N() + len(model[category].keys()))
                # results[category] += log(model[category].get(token, 0)/(len(model[category].keys()) +  len(model['neg'].keys())), 10)
            if token in model[category]:
                results[category] += log(model[category].get(token, 0)/(len(model[category].keys())), 10)

    return results

def create_model(training_set):
    model_list = []
    for category in training_set :
        files = training_set[category]
        for name in files :
            word_list = movie_reviews.words(name)
            for word in word_list :
                model_list.append((word, category))
    return model_list

def question3():
    fileids = movie_reviews.fileids()
    pos = [item for item in fileids if item.startswith('pos')]
    neg = [item for item in fileids if item.startswith('neg')]
    tset = {'pos' : pos, 'neg' : neg}


    model, total_words = train(tset)
    incorrect, correct, total = 0, 0, 0

    for fileid in fileids :
        result = classify(model, movie_reviews.words(fileid))
        true_val = fileid[0:3]
        predicted = 'na'
        if result['pos'] > result['neg'] :
            predicted = 'pos'
        else :
            predicted = 'neg'
        if predicted != true_val :
            incorrect += 1
        else:
            correct += 1
        total += 1
    print(f'Accuracy with feature selection, without k-fold: {correct/total}')

    apply_transformation = False
    model, total_words = train(tset, apply_transformation)
    incorrect, correct, total = 0, 0, 0
    for fileid in fileids :
        result = classify(model, movie_reviews.words(fileid), apply_transformation)
        true_val = fileid[0:3]
        predicted = 'na'
        if result['pos'] > result['neg'] :
            predicted = 'pos'
        else :
            predicted = 'neg'
        if predicted != true_val :
            incorrect += 1
        else:
            correct += 1
        total += 1
    print(f'Accuracy without feature selection, without k-fold: {correct/total}')

    #  Compare the performance of your classifier with and without feature
    # selection on the movie_reviews data using five-fold cross-validation.

    # kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # incorrect, correct, total = 0, 0, 0

    # kfold_accuracy_without_transformation = []
    # kfold_accuracy_with_transformation = []

    # for train_index, test_index in kfold.split(fileids):
    #     test_set = [fileids[i] for i in test_index]
    #     train_set = [fileids[i] for i in train_index]

    #     tset = {'pos' : [item for item in train_set if item.startswith('pos')], 'neg' : [item for item in train_set if item.startswith('neg')]}
    #     model = train(tset)
    #     for fileid in test_set :
    #         result = classify(model, movie_reviews.words(fileid))
    #         true_val = fileid[0:3]
    #         predicted = 'na'
    #         if result['pos'] > result['neg'] :
    #             predicted = 'pos'
    #         else :
    #             predicted = 'neg'
    #         if predicted != true_val :
    #             incorrect += 1
    #         else:
    #             correct += 1
    #         total += 1
    #     kfold_accuracy_with_transformation.append(correct/total)


    # for train_index, test_index in kfold.split(fileids):
    #     test_set = [fileids[i] for i in test_index]
    #     train_set = [fileids[i] for i in train_index]

    #     tset = {'pos' : [item for item in train_set if item.startswith('pos')], 'neg' : [item for item in train_set if item.startswith('neg')]}
    #     model = train(tset, False)
    #     for fileid in test_set :
    #         result = classify(model, movie_reviews.words(fileid), False)
    #         true_val = fileid[0:3]
    #         predicted = 'na'
    #         if result['pos'] > result['neg'] :
    #             predicted = 'pos'
    #         else :
    #             predicted = 'neg'
    #         if predicted != true_val :
    #             incorrect += 1
    #         else:
    #             correct += 1
    #         total += 1
    #     kfold_accuracy_without_transformation.append(correct/total)

    # print(f'Accuracy with feature selection, with k-fold: {sum(kfold_accuracy_with_transformation)/len(kfold_accuracy_with_transformation)}')
    # print(f'Accuracy without feature selection, with k-fold: {sum(kfold_accuracy_without_transformation)/len(kfold_accuracy_without_transformation)}')



    # compare this to the NLTK implementation of Naive Bayes using five-fold cross-validation.

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    kfold_accuracy_without_transformation = []
    kfold_accuracy_with_transformation = []

    for train_split, test_split in kfold.split(fileids):
        for train_split_num in train_split:
            train_set = [fileids[i] for i in train_split]
        for test_split_num in test_split:
            test_set = [fileids[i] for i in test_split]


        tset = {'pos' : [], 'neg' : []}

        for train_set_file in train_set:
            if train_set_file.startswith('pos'):
                tset['pos'].append(train_set_file)
            else:
                tset['neg'].append(train_set_file)

        model, _ = train(tset)

        positive_features = []

        for word, category in model['pos'].items():
            positive_features.append(({word: category}, 'pos'))

        negative_features = []

        for word, category in model['neg'].items():
            negative_features.append(({word: category}, 'neg'))

        test_features = positive_features[0:100] + negative_features[0:100]
        train_features = positive_features[100:] + negative_features[100:]

        classifier = NaiveBayesClassifier.train(train_features)

        kfold_accuracy_with_transformation.append(accuracy(classifier, test_features))

    print(f'Accuracy with feature selection: {sum(kfold_accuracy_with_transformation)/len(kfold_accuracy_with_transformation)}')


    for train_split, test_split in kfold.split(fileids):
        for train_split_num in train_split:
            train_set = [fileids[i] for i in train_split]
        for test_split_num in test_split:
            test_set = [fileids[i] for i in test_split]

        tset = {'pos' : [], 'neg' : []}

        for train_set_file in train_set:
            if train_set_file.startswith('pos'):
                tset['pos'].append(train_set_file)
            else:
                tset['neg'].append(train_set_file)

        model, total_words = train(tset, False)

        positive_features = []

        for word, category in model['pos'].items():
            positive_features.append(({word: category}, 'pos'))

        negative_features = []

        for word, category in model['neg'].items():
            negative_features.append(({word: category}, 'neg'))

        test_features = positive_features[0:100] + negative_features[0:100]
        train_features = positive_features[100:] + negative_features[100:]

        classifier = NaiveBayesClassifier.train(train_features)
        kfold_accuracy_without_transformation.append(accuracy(classifier, test_features))

    print(f'Accuracy without feature selection: {sum(kfold_accuracy_without_transformation)/len(kfold_accuracy_without_transformation)}')

## You will need to extend this to do five-fold cross-validation, and also compute accuracy.
if __name__ == "__main__" :
    question3()