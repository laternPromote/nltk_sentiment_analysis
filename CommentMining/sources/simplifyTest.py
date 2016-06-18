__author__ = 'geyan'
import simplify

with open('big5.txt', 'r') as f:
    print f.read()
simplify.tradToSimp('big5.txt')
