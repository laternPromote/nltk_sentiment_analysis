def train_brill_tagger(train_data):
    # Modules for creating the templates.
    from nltk.tag import UnigramTagger
    from nltk.tag.brill import SymmetricProximateTokensTemplate, ProximateTokensTemplate
    from nltk.tag.brill import ProximateTagsRule, ProximateWordsRule
    # The brill tagger module in NLTK.
    from nltk.tag.brill import FastBrillTaggerTrainer
    unigram_tagger = UnigramTagger(train_data)
    templates = [SymmetricProximateTokensTemplate(ProximateTagsRule, (1, 1)),
                 SymmetricProximateTokensTemplate(ProximateTagsRule, (2, 2)),
                 SymmetricProximateTokensTemplate(ProximateTagsRule, (1, 2)),
                 SymmetricProximateTokensTemplate(ProximateTagsRule, (1, 3)),
                 SymmetricProximateTokensTemplate(ProximateWordsRule, (1, 1)),
                 SymmetricProximateTokensTemplate(ProximateWordsRule, (2, 2)),
                 SymmetricProximateTokensTemplate(ProximateWordsRule, (1, 2)),
                 SymmetricProximateTokensTemplate(ProximateWordsRule, (1, 3)),
                 ProximateTokensTemplate(ProximateTagsRule, (-1, -1), (1, 1)),
                 ProximateTokensTemplate(ProximateWordsRule, (-1, -1), (1, 1))]

    trainer = FastBrillTaggerTrainer(initial_tagger=unigram_tagger,
                                     templates=templates, trace=3,
                                     deterministic=True)
    brill_tagger = trainer.train(train_data, max_rules=10)
    print
    return brill_tagger


# To train and test using Brown Corpus.
from nltk.corpus import brown

brown_train = list(brown.tagged_sents(categories='news')[:500])
brown_test = list(brown.tagged_sents(categories='news')[500:600])
brown501 = brown.tagged_sents(categories='news')[501]

bt = train_brill_tagger(brown_train)

# To tag one sentence.
sent = ['This', 'is', 'a', 'good', 'movie']
print bt.tag(sent)

# To evaluate tagger.
print 'Accuracy of Brill Tagger:', bt.evaluate(brown_test)
