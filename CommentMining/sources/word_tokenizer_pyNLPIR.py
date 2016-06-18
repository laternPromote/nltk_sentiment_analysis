# -*- coding: utf-8 -*-
__author__ = 'geyan'
import codecs

import pynlpir


def tokenize(file):
    words = []
    pynlpir.open()
    directory = '\\resources\\original files\\htl_del_4000\\'
    posWords = codecs.open(directory + file + 'Words.txt', 'w+', 'utf-8')
    with codecs.open(directory + file + '.txt', 'r', 'utf-8') as posFile:
        for s in posFile.readlines():
            # print posFile.readline()
            a = pynlpir.segment(s, pos_tagging=False)
            # print a
            for i in range(len(a)):
                # print a[i]
                if i != (len(a) - 1):
                    # print 'i='+str(i)
                    # print 'a='+str(len(a))
                    posWords.write(a[i] + ' ')
                else:
                    posWords.write(a[i] + '\r')
                    # for i in a:
                    #    posWords.write(i + ';')
                    # posWords.write('\0')
    posWords.close()

#            posWords.write('\n')
#        sentences = []
#        for line in posFile.nextline():
#            sentences.append(line)
#            print line
#        for s in sentences:
#            words.append(pynlpir.segment(s, pos_tagging=False))
#    # print words
#    with codecs.open(directory + file + 'Words.txt', 'w+', 'utf-8') as posWords:
#        for word in words:
#            for i in word:
#                posWords.write(i + ' ')
#            posWords.write('\n')
