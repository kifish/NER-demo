import re
from pyhanlp import *

def get_sents(path):
    #'../data/train.txt';'../data/dev.txt'
    sentences = []
    sentence = []
    cnt = 0
    split_pattern = re.compile(r',|\.|;|，|。|；|\?|\!|\.\.\.\.\.\.|……')
    with open(path,'r',encoding = 'utf8') as f:
        for line in f.readlines():#每行为一个字符和其tag，中间用tab隔开
            line = line.strip().split('\t')
            if(not line or len(line) < 2): continue
            word_unit = [line[0],line[1]]
            if split_pattern.match(word_unit[0]):
                sentence.append(word_unit)
                sent = ''.join((word_unit[0] for word_unit in sentence))
                nature_list = []
                for term in HanLP.segment(sent):
                    for i in range(len(term.word)):# 分词
                        nature = '{}'.format(term.nature)
                        nature_list.append(nature)
                for idx,word_unit in enumerate(sentence):
                    word_unit.insert(1,nature_list[idx]) # insert损失一些性能
                sentences.append(sentence.copy())
                sentence.clear()
            else:
                sentence.append(word_unit)
        if(len(sentence)):
            sent = ''.join((word_unit[0] for word_unit in sentence))
            nature_list = []
            for term in HanLP.segment(sent):
                for i in range(len(term.word)):  # 分词
                    nature = '{}'.format(term.nature)
                    nature_list.append(nature)
            for idx, word_unit in enumerate(sentence):
                word_unit.insert(1, nature_list[idx])  # insert损失一些性能
            sentences.append(sentence.copy())
            sentence.clear()
    return sentences

train_sentences = get_sents('../data/train.txt')
print(len(train_sentences))
print(train_sentences[0])

