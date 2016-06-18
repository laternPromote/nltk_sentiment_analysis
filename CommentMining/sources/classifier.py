import itertools
import math
import re
import string
from collections import defaultdict

import nltk
import nltk.classify.util
import nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.collocations import BigramCollocationFinder
from nltk.corpus import brown
from nltk.corpus import stopwords
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

from replacers import AntonymReplacer

directory = '\\resources\\original files\\htl_del_4000\\'
stop = stopwords.words('english')


def evaluate_features(feature_select):
    posSentences = open(directory + 'pos.txt', 'r')
    negSentences = open(directory + 'neg.txt', 'r')

    posSentences = re.split(r'\n', posSentences.read().translate(None, string.punctuation))
    negSentences = re.split(r'\n', negSentences.read().translate(None, string.punctuation))

    posFeatures = []
    negFeatures = []

    for i in posSentences:
        posWords = re.findall(r"[\w']+|[.,!?;]", i)
        posWords = [feature_select(posWords), 'pos']
        posFeatures.append(posWords)
    for i in negSentences:
        negWords = re.findall(r"[\w']+|[.,!?;]", i)
        negWords = [feature_select(negWords), 'neg']
        negFeatures.append(negWords)

    posCutoff = int(math.floor(len(posFeatures) * 9 / 10))
    negCutoff = int(math.floor(len(negFeatures) * 9 / 10))
    trainFeatures = posFeatures[:posCutoff] + negFeatures[:negCutoff]
    testFeatures = posFeatures[posCutoff:] + negFeatures[negCutoff:]

    # print len(posSentences), len(posFeatures)

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
    print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
    print 'pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos'])
    print 'pos recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos'])
    print 'neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg'])
    print 'neg recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg'])
    classifier.show_most_informative_features(10)

    return accuracy


def evaluate_without_negations(feature_select):
    replacer = AntonymReplacer()

    posSentences = open('\\resources\\original files\\pos.txt', 'r')
    negSentences = open('\\resources\\original files\\neg.txt', 'r')

    posSentences = re.split(r'\n', posSentences.read().translate(None, string.punctuation))
    negSentences = re.split(r'\n', negSentences.read().translate(None, string.punctuation))

    posFeatures = []
    negFeatures = []

    for i in posSentences:
        posWords = re.findall(r"[\w']+|[.,!?;]", i)
        posWords = replacer.replace_negations(posWords)
        posWords = [feature_select(posWords), 'pos']
        posFeatures.append(posWords)
    for i in negSentences:
        negWords = re.findall(r"[\w']+|[.,!?;]", i)
        negWords = replacer.replace_negations(negWords)
        negWords = [feature_select(negWords), 'neg']
        negFeatures.append(negWords)

    posCutoff = int(math.floor(len(posFeatures) * 9 / 10))
    negCutoff = int(math.floor(len(negFeatures) * 9 / 10))
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
    print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
    print 'pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos'])
    print 'pos recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos'])
    print 'neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg'])
    print 'neg recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg'])
    classifier.show_most_informative_features(10)

    return accuracy


def evaluate_with_adjs(feature_select):
    brown_train = list(brown.tagged_sents(categories='news'))
    # bt = train_brill_tagger(brown_train)

    posSentences = open('\\resources\\original files\\pos.txt', 'r')
    negSentences = open('\\resources\\original files\\neg.txt', 'r')

    posSentences = re.split(r'\n', posSentences.read().translate(None, string.punctuation))
    negSentences = re.split(r'\n', negSentences.read().translate(None, string.punctuation))

    posFeatures = []
    negFeatures = []

    for i in posSentences:
        posWords = re.findall(r"[\w']+|[.,!?;]", i)
        posWords = [feature_select(posWords), 'pos']
        posFeatures.append(posWords)
    for i in negSentences:
        negWords = re.findall(r"[\w']+|[.,!?;]", i)
        negWords = [feature_select(negWords), 'neg']
        negFeatures.append(negWords)

    posCutoff = int(math.floor(len(posFeatures) * 9 / 10))
    negCutoff = int(math.floor(len(negFeatures) * 9 / 10))
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
    print 'train on %d instances, test on %d instances' % (len(trainFeatures), len(testFeatures))
    print 'accuracy:', nltk.classify.util.accuracy(classifier, testFeatures)
    print 'pos precision:', nltk.metrics.precision(referenceSets['pos'], testSets['pos'])
    print 'pos recall:', nltk.metrics.recall(referenceSets['pos'], testSets['pos'])
    print 'neg precision:', nltk.metrics.precision(referenceSets['neg'], testSets['neg'])
    print 'neg recall:', nltk.metrics.recall(referenceSets['neg'], testSets['neg'])
    classifier.show_most_informative_features(10)

    return accuracy


def create_word_scores():
    posSentences = open('\\resources\\original files\\pos.txt', 'r')
    negSentences = open('\\resources\\original files\\neg.txt', 'r')
    posSentences = re.split(r'\n', posSentences.read().translate(None, string.punctuation))
    negSentences = re.split(r'\n', negSentences.read().translate(None, string.punctuation))

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
        word_fd.inc(word.lower())
        cond_word_fd['pos'].inc(word.lower())
    for word in negWords:
        word_fd.inc(word.lower())
        cond_word_fd['neg'].inc(word.lower())

    pos_word_count = cond_word_fd['pos'].N()
    neg_word_count = cond_word_fd['neg'].N()
    total_word_count = pos_word_count + neg_word_count

    word_scores = {}
    for word, freq in word_fd.iteritems():
        pos_score = BigramAssocMeasures.chi_sq(cond_word_fd['pos'][word], (freq, pos_word_count), total_word_count)
        neg_score = BigramAssocMeasures.chi_sq(cond_word_fd['neg'][word], (freq, neg_word_count), total_word_count)
        word_scores[word] = pos_score + neg_score

    return word_scores


def make_full_dict(words):
    return dict([(word, True) for word in words])


def without_stopwords(words):
    return dict([(word, True) for word in words if word not in stop])


# Run a classifier with all words as features
print 'using all words as features'
evaluate_features(make_full_dict)
# Run a classifier without negations as features
print 'using all words but negations as features'
evaluate_without_negations(make_full_dict)
# Run a classifier without stopwords as features
print 'using words except stopwords as features'
evaluate_features(without_stopwords)

word_scores = create_word_scores()


def find_best_words(word_scores, number):
    best_vals = sorted(word_scores.iteritems(), key=lambda (w, s): s, reverse=True)[:number]
    best_words = set([w for w, s in best_vals])
    return best_words


def best_word_features(words):
    return dict([(word, True) for word in words if word in best_words and word not in stop])


numbers_to_test = range(20000)
numbers_to_test = numbers_to_test[::1000]

# using n words with most information gain as features
# finding best n value
posAccuracy = []
negAccuracy = []
for num in numbers_to_test:
    print 'evaluating best %d word features' % (num)
    best_words = find_best_words(word_scores, num)
    posAccuracy.append(evaluate_features(best_word_features))
    negAccuracy.append(evaluate_without_negations(best_word_features))


# plt.axis([0, 20000, 70, 90])
# plt.plot(numbers_to_test, posAccuracy, 'ro', numbers_to_test, negAccuracy, 'bs')
# plt.show()


def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])


print 'using all words and 200 significant bigrams as features'
evaluate_features(bigram_word_feats)
