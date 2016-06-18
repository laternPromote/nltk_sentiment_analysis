from nltk.corpus import wordnet


class AntonymReplacer(object):
    def replace(self, word, pos=None):
        antonyms = []
        for syn in wordnet.synsets(word, pos=pos):
            for lemma in syn.lemmas:
                for antonym in lemma.antonyms():
                    antonyms.append(antonym.name)
        if len(antonyms) != 0:
            return antonyms[0]
        else:
            return None

    def replace_negations(self, sent):
        i, l = 0, len(sent)
        words = []
        while i < l:
            word = sent[i]
            if word == 'not' and i + 1 < l:
                ant = self.replace(sent[i + 1])
                if ant:
                    words.append(ant)
                    i += 2
                    continue
            words.append(word)
            i += 1
        return words
