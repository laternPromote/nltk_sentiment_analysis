# -*- coding: UTF-8 -*-
__author__ = 'geyan'

from ctypes import *

dll = CDLL('NLPIR.dll')


def fillprototype(f, restype, argtypes):
    f.restype = restype
    f.argtypes = argtypes


MY_NLPIR_Init = getattr(dll, 'NLPIR_Init')
MY_NLPIR_Exit = getattr(dll, 'NLPIR_Exit')
MY_NLPIR_ParagraphProcess = getattr(dll, 'NLPIR_ParagraphProcess')
MY_NLPIR_ImportUserDict = getattr(dll, 'NLPIR_ImportUserDict')
MY_NLPIR_FileProcess = getattr(dll, 'NLPIR_FileProcess')
MY_NLPIR_AddUserWord = getattr(dll, 'NLPIR_AddUserWord')
MY_NLPIR_SaveTheUsrDic = getattr(dll, 'NLPIR_SaveTheUsrDic')
MY_NLPIR_DelUsrWord = getattr(dll, 'NLPIR_DelUsrWord')
MY_NLPIR_GetKeyWords = getattr(dll, 'NLPIR_GetKeyWords')
MY_NLPIR_GetFileKeyWords = getattr(dll, 'NLPIR_GetFileKeyWords')
MY_NLPIR_GetNewWords = getattr(dll, 'NLPIR_GetNewWords')
MY_NLPIR_GetFileNewWords = getattr(dll, 'NLPIR_GetFileNewWords')
MY_NLPIR_SetPOSmap = getattr(dll, 'NLPIR_SetPOSmap')
MY_NLPIR_FingerPrint = getattr(dll, 'NLPIR_FingerPrint')
# New Word Identification
MY_NLPIR_NWI_Start = getattr(dll, 'NLPIR_NWI_Start')
MY_NLPIR_NWI_AddFile = getattr(dll, 'NLPIR_NWI_AddFile')
MY_NLPIR_NWI_AddMem = getattr(dll, 'NLPIR_NWI_AddMem')
MY_NLPIR_NWI_Complete = getattr(dll, 'NLPIR_NWI_Complete')
MY_NLPIR_NWI_GetResult = getattr(dll, 'NLPIR_NWI_GetResult')
MY_NLPIR_NWI_Result2UserDict = getattr(dll, 'NLPIR_NWI_Result2UserDict')

fillprototype(MY_NLPIR_Init, c_bool, [c_char_p, c_int])
fillprototype(MY_NLPIR_Exit, c_bool, None)
fillprototype(MY_NLPIR_ParagraphProcess, c_char_p, [c_char_p, c_int])
fillprototype(MY_NLPIR_ImportUserDict, c_uint, [c_char_p])
fillprototype(MY_NLPIR_FileProcess, c_double, [c_char_p, c_char_p, c_int])
fillprototype(MY_NLPIR_AddUserWord, c_int, [c_char_p])
fillprototype(MY_NLPIR_SaveTheUsrDic, c_int, None)
fillprototype(MY_NLPIR_DelUsrWord, c_int, [c_char_p])
fillprototype(MY_NLPIR_GetKeyWords, c_char_p, [c_char_p, c_int, c_bool])
fillprototype(MY_NLPIR_GetFileKeyWords, c_char_p, [c_char_p, c_int, c_bool])
fillprototype(MY_NLPIR_GetNewWords, c_char_p, [c_char_p, c_int, c_bool])
fillprototype(MY_NLPIR_GetFileNewWords, c_char_p, [c_char_p, c_int, c_bool])
fillprototype(MY_NLPIR_SetPOSmap, c_int, [c_int])
fillprototype(MY_NLPIR_FingerPrint, c_ulong, [c_char_p])
# New Word Identification
fillprototype(MY_NLPIR_NWI_Start, c_bool, None)
fillprototype(MY_NLPIR_NWI_AddFile, c_bool, [c_char_p])
fillprototype(MY_NLPIR_NWI_AddMem, c_bool, [c_char_p])
fillprototype(MY_NLPIR_NWI_Complete, c_bool, None)
fillprototype(MY_NLPIR_NWI_GetResult, c_char_p, [c_int])
fillprototype(MY_NLPIR_NWI_Result2UserDict, c_uint, None)

# 初始化分词系统
if not MY_NLPIR_Init('', 1):
    print "Initial fail"
    exit()

posResults = []
negResults = []
directory = '\\resources\\original files\\htl_del_4000\\'


def tokenizeTextFile(arg):
    with open(directory + arg + '.txt', 'r') as f:
        for s in f:
            negResults.append(MY_NLPIR_ParagraphProcess(s, c_int(3)))
        with open(directory + arg + 'Tokenize.txt', 'w+') as negTokenize:
            for negResult in negResults:
                negTokenize.write(negResult)


tokenizeTextFile('neg')
tokenizeTextFile('pos')

MY_NLPIR_Exit()
