import numpy as np
from collections import defaultdict
import math
import re
from functools import reduce

class HMM:
    """
    usage:
    from src.hmm import HMM
    h = HMM()
    """
    def __init__(self):
        self.hidden_states = ['B-LOC', 'B-ORG', 'B-PER', 'I-PER', 'I-ORG', 'I-LOC','O']
        # self.observation = the set of all words in training samples
        self.init_prob = self.load_init_prob()
        self.transmission_matrix = self.load_tran_prob()
        self.emission_matrix = self.load_emit_prob()
    def load_init_prob(self):
        d = {}
        with open('../data/initial_vector.txt') as f:
            for line in f.readlines():
                state,prob = line.strip().split('\t')
                prob = eval(prob)
                d[state] = prob
        valid_d = {} # 有些状态不可能成为初始状态;其实NER中, 不一定需要用到初始状态的分布
        for k,v in d.items():
            if k[0] == 'I':
                continue
            valid_d[k] = v

        sum_val = sum(valid_d.values())
        for k in valid_d.keys():
            valid_d[k] /= sum_val
        return valid_d

    def load_tran_prob(self):
        d = {}
        with open('../data/tran_with_offset.txt') as f:
            for line in f.readlines():
                tran,prob = line.strip().split('\t')
                state1,state2 = tran.split('->')
                prob = eval(prob)
                d[(state1,state2)] = math.log(prob)
        return d
    def load_emit_prob(self):
        """
        在nlp序列任务中，发射概率是指在某个标注下，生成某个word的概率。
        viterbi解码需要某label下生成给定word的概率值:
        P(word|B-LOC) = ...
        P(word|B-ORG) = ...
        这里有问题：
        举例来说, 对'我来到北京天安门'做NER
        不进行概率平滑，由于P(word|O) 很大，且P(word|state) 很容易出现0，如果用乘法，会直接导致变为0
        导致最终的标记序列为['O','O','O','O','O','O','O','O']
        故采用 +1 平滑，见count

        即使这样还不行，由于OO转移的概率相较于其他状态转移概率非常非常高，导致非常容易出现
        ['O','O','O','O','O','O','O','O',......]
        必须要压制OO的转移概率
        """
        d = {}
        with open('../data/emit.txt') as f:
            for line in f.readlines():
                line = line.strip().split(' ',maxsplit= 2)
                state, oov_prob = line[0],line[1]
                oov_prob = eval(oov_prob)
                oov_prob = math.log(oov_prob)
                word2prob = defaultdict(lambda : oov_prob)
                for item in line[2].split(';'):
                    word, prob = item.split(':')
                    if word == '':
                        word = ';' # fix ;
                    prob = eval(prob)
                    prob = math.log(prob)
                    word2prob[word] = prob
                d[state] = word2prob
        return d
    def label_sentence(self,sentence):
        seq_nodes = []
        for word in sentence:
            node = {}
            for state in self.hidden_states:
                node[state] = self.emission_matrix[state][word]
            seq_nodes.append(node)
        return seq_nodes


    def label_text(self,sents):
        labels = []
        for sent in sents:
            labels = labels.extend(self.recognize_seq(sent))
        return labels

    def viterbi(self,nodes):
        trans_prob = self.transmission_matrix
        #优化初始概率，硬解码
        paths = [ ([state],nodes[0][state])  for state in self.hidden_states if state[0] != 'I' ]
        # print(paths)

        for node_idx in range(1, len(nodes)):
            paths_ = paths
            paths = []
            for label in nodes[node_idx].keys():
                # 内循环 求以给定的label(hidden state)作为最后一个label的最优路径
                paths_endwith_this_label = []
                for path_and_prob in paths_:
                    # print(path_and_prob)
                    # print()
                    path, prob = path_and_prob
                    path = path.copy()
                    pre_node_label = path[-1]
                    if (pre_node_label, label) in trans_prob.keys():
                        # 把概率相乘等价地转化为相加
                        prob_endwith_this_label = prob + trans_prob[(pre_node_label,label)] + nodes[node_idx][label]
                        path.append(label)
                        paths_endwith_this_label.append((path, prob_endwith_this_label))
                max_idx = 0
                for idx in range(1,len(paths_endwith_this_label)):
                    if paths_endwith_this_label[idx][1] > paths_endwith_this_label[max_idx][1]:
                        max_idx = idx
                paths.append(paths_endwith_this_label[max_idx])
        max_idx = 0
        for idx in range(1, len(paths)):
            if paths[idx][1] > paths[max_idx][1]:
                max_idx = idx
        # print(paths)
        return paths[max_idx][0]

    def recognize_seq(self,sentence,verbose = False):
        seq_nodes = self.label_sentence(sentence)
        path = self.viterbi(seq_nodes)
        if verbose:
            print(path)
            for idx in range(len(path)):
                print(sentence[idx] + '\t' + path[idx])
        return path

    def get_sents(self,path):
        # '../data/train.txt';'../data/dev.txt'
        sentences = []
        sentence = []
        split_pattern = re.compile(r',|\.|;|，|。|；')  # .要转义，不然表示的是通配符
        with open(path, 'r', encoding='utf8') as f:
            for line in f.readlines():  # 每行为一个字符和其tag，中间用tab隔开
                line = line.strip().split('\t')
                if (not line or len(line) < 2): continue
                word = line[0]
                if split_pattern.match(word):
                    sentence.append(word)
                    sentences.append(sentence.copy())
                    sentence.clear()
                else:
                    sentence.append(word)
            if (len(sentence)):
                sentences.append(sentence.copy())
                sentence.clear()
        return sentences

    def recognize_text(self,text_path = '../data/dev.txt'):
        labels = []
        sents = self.get_sents(text_path)
        text = reduce(lambda x,y : x + y, sents)
        for sent in sents:
            labels.extend(self.recognize_seq(sent))
        res = zip(text,labels)
        with open('../data/pred.txt','w',encoding='utf8') as f:
            for item in res:
                f.write(item[0] + '\t' + item[1] + '\n')
        print('completed recognition')

if __name__ == '__main__':
    h = HMM()
    # print(h.label_sentence('我来到北京天安门'))
    # print(h.recognize_seq('我来到北京天安门'))
    # print(h.recognize_seq('中共中央'))
    # print(h.recognize_seq('北大清华图书馆'))
    h.recognize_text()



