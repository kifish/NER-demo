import re,os
from collections import defaultdict
import numpy as np
from functools import reduce
from sklearn_crfsuite import metrics
from seqeval.metrics import classification_report



def load_data_and_labels(path):
    split_pattern = re.compile(r'[；;。，、？！\.\?,! ]')
    x = []
    y = []
    sentence = []
    labels = []
    seq_max_len = 99 # 有些句子非常长,设定最长句子为100
    with open(path,'r',encoding = 'utf8') as f:
        for line in f.readlines(): # 每行为一个字符和其tag，中间用tab隔开
            line = line.strip().split('\t')
            if(not line or len(line) < 2): continue
            word, tag = line[0], line[1]
            tag = tag if tag != 'OO' else 'O'
            if split_pattern.match(word) or len(sentence) == seq_max_len:
                sentence.append(word)
                labels.append(tag)
                x.append(sentence.copy())
                y.append(labels.copy())
                sentence.clear()
                labels.clear()
            else:
                sentence.append(word)
                labels.append(tag)
        if(len(sentence)):
            x.append(sentence.copy())
            sentence.clear()
            y.append(labels.copy())
            labels.clear()
    return x,y

def load_data(path):
    split_pattern = re.compile(r'[；;。，、？！\.\?,! ]')
    x = []
    raw = []
    seq_max_len = 99 # 有些句子非常长,设定最长句子为100
    with open(path,'r',encoding = 'utf8') as f: # 这里是按行来分割的。
        for line in f.readlines():
            line = line.strip()
            raw.append(line)
            seqs = []
            seq = []
            for word in line:
                if split_pattern.match(word) or len(seq) == seq_max_len:
                    seq.append(word)
                    seqs.append(seq)
                    seq = []
                else:
                    seq.append(word)
            if(len(seq)):
                seqs.append(seq)
            x.append(seqs)
    x = [seq for line in x for seq in line] # 简单按句子来分割，一个句子即一个样本
    with open(os.path.join(os.path.dirname(path),'processed_test.txt'),'w',encoding='utf8') as f:
        for line in raw:
            for ch in line:
                f.write(ch + '\n')
            f.write('\n')
    return x


def load_processed_data(path = '../data/processed_test.txt'):
    x = []
    with open(path,'r',encoding='utf8') as f:
        for line in f.readlines():
            x.append(line.strip())
    return x

def save_pred(pred,save_path = '../data/val_pred.txt',for_validation = True):
    if for_validation:
        with open(save_path, 'w', encoding='utf8') as f:
            x_val, _ = load_data_and_labels('../data/dev.txt')
            x_val = reduce(lambda x,y: x + y,x_val)
            pred = reduce(lambda x,y: x + y,pred) # 2维list
            result = zip(x_val, pred)
            for item in result:
                f.write(item[0] + '\t' + item[1] + '\n')
    else: # save pred for test data
        with open(save_path, 'w', encoding='utf8') as f:
            x_test = load_processed_data()
            pred = reduce(lambda x,y: x + y,pred) # 2维list/tuple
            idx2 = 0
            for idx1 in range(len(x_test)):
                if x_test[idx1] == '':
                    f.write('\n')
                else:
                    f.write(x_test[idx1] + '\t' + pred[idx2] + '\n') # 为了保留空行
                    idx2 += 1



def eval(y_test,pred):
    labels = ['B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']
    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        y_test, pred, labels=sorted_labels, digits=3
    ))

    y_test = reduce(lambda x,y:x + y,y_test)
    pred = reduce(lambda x,y: x + y, pred) # pred内部是tuple
    pred = list(pred)
    print(classification_report(y_test,pred,digits = 4)) # 只接受list

if __name__ == '__main__':
    pass 


