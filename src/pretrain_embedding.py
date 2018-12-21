import gensim,logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from gensim.models import Word2Vec
import os
import re
from functools import reduce
def get_texts(path):
    path = os.path.abspath(path)
    texts = []
    for dir in os.listdir(path):
        dir_path = os.path.join(path, dir)
        if os.path.isdir(dir_path):
            for filename in os.listdir(dir_path):
                if filename[-4:] == '.txt':
                    texts.append(get_text(os.path.join(dir_path,filename)))
    return texts

def get_text(path):
    # 具体的格式见：https://rare-technologies.com/word2vec-tutorial/
    useful_pattern = re.compile(r'[0-9\u4E00-\u9FA5；;：:。，、？！\.\?,!\"\'\”\“\‘\’《》]+', re.M)
    text = []
    with open(path,'r',encoding = 'utf8') as f:
        for line in f.readlines():
            res = useful_pattern.findall(line)
            if res:
                seq = []
                for item in res:
                    if item != ' ': #不要空格
                        seq += list(item)
                text.append(seq) #一行一行分割之后，语料中的句子已经比较短了，不需要用标点符号再切割句子了。
    return text

def pretrain_wv():
    corpus_path = '../data/word2vec_corpus/'
    texts = get_texts(corpus_path)
    seqs = [seq for text in texts for seq in text]  # 也可以改成generator
    model = Word2Vec(size=100, window=5, min_count=1, workers=4)
    model.build_vocab(seqs)
    model.train(seqs,total_examples=model.corpus_count,epochs = 15)
    model.wv.save_word2vec_format('../data/word2vec.txt',binary=False)
pretrain_wv()