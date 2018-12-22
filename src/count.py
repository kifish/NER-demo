import os
import re
from collections import defaultdict
import sys
hidden_states = ['B-LOC', 'B-ORG', 'B-PER', 'I-PER', 'I-ORG', 'I-LOC','O']

def get_sents(path):
    sentences = []
    sentence = []
    split_pattern = re.compile(r',|\.|;|，|。|；|\?|\!|\.\.\.\.\.\.|……') #.要转义，不然表示的是通配符
    with open(path,'r',encoding = 'utf8') as f:
        for line in f.readlines():#每行为一个字符和其tag，中间用tab隔开
            line = line.strip().split('\t')
            if(not line or len(line) < 2): continue
            word_unit = (line[0],line[1]) #word and tag
            if split_pattern.match(word_unit[0]):
                sentence.append(word_unit)
                sentences.append(sentence.copy())
                sentence.clear()
            else:
                sentence.append(word_unit)
        if(len(sentence)):
            sentences.append(sentence.copy())
            sentence.clear()
    return sentences


def gen_initial_vec(hidden_states):
    d = {state : 0 for state in hidden_states}
    sents = get_sents("../data/train.txt")
    for sent in sents:
        word_and_tag = sent[0] #每个句子的第一个word
        word, tag = word_and_tag
        if tag == 'OO':  # 把'OO' 归成 'O'
            tag = 'O'
        d[tag] += 1
    total_cnt = sum(d.values())
    for state,cnt in d.items():
        d[state] = cnt / total_cnt
    with open("../data/initial_vector.txt",'w',encoding = 'utf8') as f:
        for state,fre in d.items():
            f.write(state + '\t' + str(fre)+ '\n')
    path = os.path.abspath('../data/initial_vector.txt')
    print("generated initial_vector saved in " + path)

def gen_tran_prob(hidden_states, ratio = None):
    """
    由于OO转移的概率相较于其他状态转移概率非常非常高，导致非常容易出现
    ['O','O','O','O','O','O','O','O',......]
    必须要压制OO的转移概率
    """
    d = defaultdict(int)
    tran_cnt = 0
    train_sentences = get_sents('../data/train.txt')
    for sent in train_sentences:
        idx = 0
        end = len(sent) - 1
        while idx < end:
            state1 = sent[idx][1]
            state2 = sent[idx + 1][1]
            state1 = 'O' if state1 == 'OO' else state1
            state2 = 'O' if state2 == 'OO' else state2
            d[state1 + '->' + state2] += 1
            tran_cnt += 1
            idx += 1

    valid_d = {} # 删去无效的状态转移
    for k,v in d.items():
        if v > 0:
            valid_d[k] = v / tran_cnt

    if ratio:
        valid_d['O->O'] = ratio #压制,劫富济贫,概率再分配
        left = 1 - ratio
        for k, v in valid_d.items():
            if k == 'O->O':
                continue
            valid_d[k] = v / left

    with open("../data/tran_with_offset.txt",'w',encoding = 'utf8') as f:
        for k,v in valid_d.items():
            f.write(k + '\t' + str(v) + '\n')
    path = os.path.abspath('../data/tran_with_offset.txt')
    print("generated transition_probability with offset saved in " + path)

def gen_emit_prob(hidden_states):
    """
    训练样本中并不能包括所有的词，也没法确定所有词的总数量，因此这里的平滑采用了折衷的方案
    """
    d = {state : defaultdict(int) for state in hidden_states}
    with open("../data/train.txt") as f:
        for line in f.readlines():
            word_and_tag = line.strip().split('\t')
            if(len(word_and_tag) <= 1):
                continue
            word = word_and_tag[0]
            tag = word_and_tag[1]
            tag = 'O' if tag == 'OO' else tag
            d[tag][word] += 1
    for tag,word2cnt in d.items():
        cnt = sum(word2cnt.values()) + len(word2cnt.values()) + 1
        for word in word2cnt.keys():
            word2cnt[word] = (word2cnt[word] + 1) / cnt
        word2cnt['OOV'] = 1 / cnt
    with open('../data/emit.txt','w',encoding='utf8') as f:
        for tag,word2cnt in d.items():
            s = []
            for word,fre in word2cnt.items():
                s.append(word + ':' + str(fre))
            s = ';'.join(s)
            f.write(tag + ' ' + str(word2cnt['OOV']) + ' ' + s + '\n')
    path = os.path.abspath('../data/emit.txt')
    print("generated emission_probability saved in " + path)


if __name__ == "__main__":
    gen_initial_vec(hidden_states)
    if len(sys.argv) >= 2:
        gen_tran_prob(hidden_states, ratio = eval(sys.argv[1]))
    else:
        gen_tran_prob(hidden_states)
    gen_emit_prob(hidden_states)




