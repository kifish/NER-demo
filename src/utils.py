# from anago.utils import load_data_and_labels  #这里是靠空行来切割句子的，这会导致句子过于长。
import re
from collections import defaultdict
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from functools import reduce

def load_data_and_labels(path):
    split_pattern = re.compile(r'[；;。，、？！\.\?,! ]')
    x = []
    y = []
    sentence = []
    labels = []
    seq_max_len = 99 #有些句子非常长,设定最长句子为100
    with open(path,'r',encoding = 'utf8') as f:
        for line in f.readlines():#每行为一个字符和其tag，中间用tab隔开
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

def save_pred(pred):
    with open('../data/pred.txt', 'w', encoding='utf8') as f:
        x_test, _ = load_data_and_labels('../data/dev.txt')
        x_test = reduce(lambda x,y: x + y,x_test)
        pred = reduce(lambda x,y: x + y,pred) # 2维list
        result = zip(x_test, pred)
        for item in result:
            f.write(item[0] + '\t' + item[1] + '\n')

class transformer_x():
    def __init__(self):
        self.word2id = defaultdict(lambda : 1) #OOV 是1。无word是0，其他为对应的idx
        self.max_len = None
        self.vocab_size = None

    def fit(self,X):
        all_words = [word for sent in X for word in sent]
        self.max_len = max(map(len,X))
        words = set(all_words)
        self.vocab_size = len(words) + 2 # padding ; OOV
        for index,word in enumerate(words):
            self.word2id[word] = index + 2

        X = [[self.word2id.get(word,1) for word in x] for x in X ]
        X = pad_sequences(X,maxlen = self.max_len)
        return X
    def tran(self,X):
        X = [[self.word2id.get(word,1) for word in x] for x in X ]
        X = pad_sequences(X,maxlen = self.max_len)
        return X

class transformer_y():
    def __init__(self,max_len):
        self.max_seq_len = max_len
        self.tags  = ['B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER','O']
        self.tag2id = {tag : idx + 1 for idx,tag in enumerate(self.tags)}
        self.id2tag = {idx + 1 : tag for idx,tag in enumerate(self.tags)}
        self.tag2id['padding'] = 0 #要加padding tag，因为 x输入的时候对不足部分补0了，所以y也需要对不足部分补0
        self.id2tag[0] = 'padding'
    def to_onehot(self,Y):
        res = []
        for y in Y:
            seq = []
            for tag in y:
                seq.append(self.tag2id.get(tag,7))
            seq_len = len(seq)
            if seq_len > self.max_seq_len:
                seq = seq[-100:]
            elif seq_len < self.max_seq_len:
                seq = [0] * (self.max_seq_len - seq_len) + seq
            seq = to_categorical(seq,num_classes= 8)
            res.append(seq)
        Y = res
        Y = np.asarray(Y)
        return Y
    def to_tag(self,Y):
        res = []
        for y in Y:
            seq_tag = []
            for tag_onehot in y:
                tag = np.where(tag_onehot == 1) #numpy.ndarray
                # 注意 np.where 和list.index 不一样。 前者返回的是一个tuple
                tag = tag[0][0]
                if tag == 0: #padding
                    continue
                tag = self.id2tag[tag]
                seq_tag.append(tag)
            res.append(seq_tag)
        return res


if __name__ == '__main__':
    x_train, y_train = load_data_and_labels('../data/train.txt')
    # x_test, y_test = load_data_and_labels('../data/dev.txt')
    # max_idx = 0
    # for idx,seq in enumerate(x_train):
    #     if len(seq) > len(x_train[max_idx]):
    #         max_idx = idx
    # print(x_train[max_idx])
    # #有些句子特别长
    # from collections import defaultdict
    # d = defaultdict(int)
    # for seq in x_train:
    #     d[len(seq)] += 1
    # print(d)

    x_train_sub = x_train[:3]
    transformer_x = transformer_x()
    print(transformer_x.fit(x_train_sub))
    y_train_sub = y_train[:3]
    transformer_y = transformer_y(transformer_x.max_len)
    print(transformer_y.to_onehot(y_train_sub))




