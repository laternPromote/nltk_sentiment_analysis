# -*- coding: utf-8 -*-
__author__ = 'geyan'
import codecs
import collections
import itertools
import math
import os.path
import re
from collections import defaultdict

import nltk
import nltk.classify.util
import nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist


# 将dict中的中文转换成utf-8的格式，以实现可以在控制台输出中文
def convert(data):
    if isinstance(data, basestring):
        return unicode(data, 'utf-8')
    elif isinstance(data, collections.Mapping):
        return dict(map(convert, data.items()))
    elif isinstance(data, collections.Iterable):
        return type(data)(map(convert, data))
    else:
        return data


directory = '\\resources\\original files\\htl_del_4000\\'
negDirectory = '\\resources\\original files\\htl_del_4000\\neg\\'
posDirectory = '\\resources\\original files\\htl_del_4000\\pos\\'

"""
evaluate_features(feature_select)
该函数用来进行对数据集分类的正确率验证
参数：feature_select为一个函数，用来提供不同的feature选择方案
"""


def evaluate_features(feature_select):
    posFile = directory + 'posWords.txt'
    negFile = directory + 'negWords.txt'
    if os.path.exists(posFile):
        posSentences = codecs.open(posFile, 'r', 'utf-8')
    else:
        print("posFile doesn't exist")
    if os.path.exists(negFile):
        negSentences = codecs.open(negFile, 'r', 'utf-8')
    else:
        print("negFile doesn't exist")

    # each line is a single comment
    posSentences = re.split(r'\n', posSentences.read())
    # print posSentences
    negSentences = re.split(r'\n', negSentences.read())

    posFeatures = []
    negFeatures = []

    # cut every single chinese word
    for i in posSentences:
        posWords = re.findall(r"[\w']+|[.,!?;]", i, re.UNICODE)
        posWords = [feature_select(posWords), u'好评']
        posFeatures.append(posWords)
    for i in negSentences:
        negWords = re.findall(r"[\w']+|[.,!?;]", i, re.UNICODE)
        negWords = [feature_select(negWords), u'差评']
        negFeatures.append(negWords)
    # print posFeatures
    posCutoff = int(math.floor(len(posFeatures) * 9 / 10))
    negCutoff = int(math.floor(len(negFeatures) * 9 / 10))
    print('sameple size: ' + str(len(posFeatures) + len(negFeatures)))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]

    # print trainFeatures

    classifier = NaiveBayesClassifier.train(trainFeatures)

    referenceSets = defaultdict(set)
    testSets = defaultdict(set)

    for i, (features, label) in enumerate(testFeatures):
        referenceSets[label].add(i)
        predicted = classifier.classify(features)
        testSets[predicted].add(i)

    # print testFeatures


    accuracy = nltk.classify.util.accuracy(classifier, testFeatures)
    print('train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures)))
    print('准确率:', math.floor((nltk.classify.util.accuracy(classifier, testFeatures) * 10 ** 4)) / 100, '%')
    print('好评 查准率:', math.floor((nltk.metrics.precision(referenceSets[u'好评'], testSets[u'好评']) * 10 ** 4)) / 100, '%')
    print('好评 查全率:', math.floor((nltk.metrics.recall(referenceSets[u'好评'], testSets[u'好评']) * 10 ** 4)) / 100, '%')
    print('差评 查准率:', math.floor((nltk.metrics.precision(referenceSets[u'差评'], testSets[u'差评']) * 10 ** 4)) / 100, '%')
    print('差评 查全率:', math.floor((nltk.metrics.recall(referenceSets[u'差评'], testSets[u'差评']) * 10 ** 4)) / 100, '%')
    sorted(classifier.labels())
    classifier.show_most_informative_features(20)
    # most_informative_features = classifier.most_informative_features(20)
    # print most_informative_features.encode('utf8')
    return accuracy


# 将所有词当作特征词
def make_full_dict(words):
    return dict([(word, u'特征词') for word in words])


# 引入bigram（双词）作为额外的特征词
def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, u'特征词') for ngram in itertools.chain(words, bigrams)])


# Run a classifier with all words as features
print('基于词袋法的分类结果')
evaluate_features(make_full_dict)

# Run a classifier with all words and most significant bigrams
print('基于n-gram的分类结果')
evaluate_features(bigram_word_feats)


def create_word_scores():
    posFile = directory + 'posWords.txt'
    negFile = directory + 'negWords.txt'
    if os.path.exists(posFile):
        posSentences = codecs.open(posFile, 'r', 'utf-8')
    else:
        print("posFile doesn't exist")
    if os.path.exists(negFile):
        negSentences = codecs.open(negFile, 'r', 'utf-8')
    else:
        print("negFile doesn't exist")

    # each line is a single comment
    posSentences = re.split(r'\n', posSentences.read())
    # print posSentences
    negSentences = re.split(r'\n', negSentences.read())

    posWords = []
    negWords = []
    for i in posSentences:
        posWord = re.findall(r"[\w']+|[.,!?;]", i)
        posWords.append(posWord)
    for i in negSentences:
        negWord = re.findall(r"[\w']+|[.,!?;]", i)
        negWords.append(negWord)
    posWords = list(itertools.chain(*posWords))
    negWords = list(itertools.chain(*negWords))

    word_fd = FreqDist()
    cond_word_fd = ConditionalFreqDist()

    for word in posWords:
        word_fd[word] += 1
        cond_word_fd['pos'][word] += 1
    for word in negWords:
        word_fd[word] += 1
        cond_word_fd['neg'][word] += 1

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.items():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores


def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.items(), key=lambda w_s: w_s[1], reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words


def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words])


    # numbers_to_test = range(20000)
    # numbers_to_test = numbers_to_test[::1000]


word_scores = create_word_scores()
best_words = find_best_words(word_scores, 1000)

print("基于信息熵的分类结果")
evaluate_features(best_word_features)


def best_bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    d = dict([(bigram, True) for bigram in bigrams])
    d.update(best_word_features(words))
    return d


print('将信息熵和n-gram结合的分类结果')
evaluate_features(best_bigram_word_feats)
