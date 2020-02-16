# -*- coding: utf-8 -*- 
# @Time : 2019/12/26 13:51 
# @Author : 01377903
# @File : Tools.py 
# @Comment :
from gensim import models
import pandas as pd

def LoadVariableLengthData(path,batchSize=50):
    '''
    从本地文件加载不定长数据（不定长的路由序列）,形如 010A-010B-020C，手动padding，并按batchSize分割
    :param path:
    :param batchSize:
    :return:
    '''
    def miniSplit(strRoute):
        res = strRoute.tolist()[0].split('-')
        return res
    rawData = pd.read_csv(path, header=None)
    # rawData.applymap(lambda x: x.split('-'))
    rawData = rawData.apply(miniSplit,axis=1)
    rawData = rawData.apply(GetSentenceVec)
    return rawData


def GetSentenceVec(inSentence, word2vecModelPath='test.mdl'):
    '''

    :param inSentence: 单个句子,list format
    :param word2vecModelPath:
    :return:lstRes, 形如[[词向量],[],...],list 的 list， 长度为句子长度
    '''
    new_model = models.Word2Vec.load(word2vecModelPath)
    lstRes = []
    keyWords = new_model.wv.vocab.keys()
    # 新词，即不在keyWords里面的词。目前不用，留作以后扩展
    lstNewWords = []
    for word in inSentence:
        if word in keyWords:
            lstRes.append(new_model[word].tolist())
        else:
            print('New word appeared')
            lstNewWords.append(word)
    return lstRes
def TestBuildCorpusVec(inFilePath, word2vecModelPath='test.mdl'):
    '''
    直接把输入和输出文本做成相应的vec
    :return:
    '''
    new_model = models.Word2Vec.load(word2vecModelPath)
    lstRes = []
    lstNewWords = []
    keyWords = new_model.wv.vocab.keys()
    try:
        fopen = open(inFilePath, 'r', encoding='UTF-8-sig')
        data = fopen.read().splitlines()

        for line in data:
            sentenceVec = []
            # 以下这句在平时要注释掉
            # if
            for word in line.split('-'):
                if word in keyWords:
                    sentenceVec.append(new_model[word].tolist())
                else:
            #         暂时当作这个词不存在，且输出出来，后续可以用于补充训练word2vec
                    lstNewWords.append(word)
            lstRes.append(sentenceVec)
    except:
        print('ERR')
    else:
        fopen.close()
    # pdRes = pd.DataFrame(lstRes)
    # pdRes.to_csv(outFilePath)
    # 先输出看一眼，没有，很好
    # print(lstNewWords)
    # dfOutFile = pd.DataFrame(lstRes)
    # dfOutFile.to_csv(outFilePath, header=False, index=False)
    # return outFilePath
    return lstRes

def BuildCorpusVec(inFilePath, outFilePath, word2vecModelPath='test.mdl'):
    '''
    直接把输入和输出文本做成相应的vec
    :return:
    '''
    new_model = models.Word2Vec.load(word2vecModelPath)
    lstRes = []
    lstNewWords = []
    keyWords = new_model.wv.vocab.keys()
    try:
        fopen = open(inFilePath, 'r', encoding='UTF-8-sig')
        data = fopen.read().splitlines()

        for line in data:
            sentenceVec = []
            strSentenceVec = ''
            # 以下这句在平时要注释掉
            # if
            for word in line.split('-'):
                if word in keyWords:
                    # sentenceVec.append(new_model[word].tolist())
                    tmp = ' [' + ','.join([str(x) for x in new_model[word].tolist()]) + ']'
                    strSentenceVec = strSentenceVec + ' [' + ','.join([str(x) for x in new_model[word].tolist()]) + ']'
                else:
            #         暂时当作这个词不存在，且输出出来，后续可以用于补充训练word2vec
                    lstNewWords.append(word)
            # lstRes.append(sentenceVec)
            fo = open(outFilePath, 'a')
            fo.write(strSentenceVec)
            fo.close()


    except:
        print('ERR')
    else:
        fopen.close()
    # pdRes = pd.DataFrame(lstRes)
    # pdRes.to_csv(outFilePath)
    # 先输出看一眼，没有，很好
    # print(lstNewWords)
    # dfOutFile = pd.DataFrame(lstRes)
    # dfOutFile.to_csv(outFilePath, header=False, index=False)
    # return outFilePath
    return None

def GetSentenceVec(sentence):
    '''

    :param sentence:
    :return: 组成
    '''
    result = []
    for word in sentence:
        new_model = models.Word2Vec.load('test.mdl')
        result.append(new_model[word])
    return result

def GetLstFromCsv(filePath):
    '''
    :param filePath: 输入文件路径
    :return: list
    '''
    result = []
    try:
        fopen = open(filePath, 'r', )
        data = fopen.read().splitlines()
        for line in data:
            # 以下这句在平时要注释掉
            # if
            result.append(line.split(',')[-1].split('-'))
    except :
        print('ERR')
    else:
        fopen.close()
    return result[1:]
