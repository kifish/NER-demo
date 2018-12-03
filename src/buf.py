# %matplotlib inline
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import sklearn_crfsuite
from itertools import chain
import nltk
import sklearn
import scipy.stats
from sklearn.metrics import make_scorer
from sklearn.cross_validation import cross_val_score
from sklearn.grid_search import RandomizedSearchCV
import re
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn_crfsuite import metrics

# %%time
def get_sents(path):
    #'./data/train.txt';'./data/dev.txt'
    sentences = []
    sentence = []
    cnt = 0
    split_pattern = re.compile(r',|\.|;|，|。|；') #.要转义，不然表示的是通配符
    with open(path,'r',encoding = 'utf8') as f:
        for line in f.readlines():#每行为一个字符和其tag，中间用tab隔开
            line = line.strip().split('\t')
            try:
                if(not line or len(line) < 2): continue
            except:
                print(line)
            if(cnt > 100):break
            word = (line[0],line[1])
            if split_pattern.match(word[0]):
                sentence.append(word)
                sentences.append(sentence.copy())
                sentence.clear()
            else:
                sentence.append(word)
        if(len(sentence)):
            sentences.append(sentence.copy())
            sentence.clear()
        cnt += 1
    return sentences
sentences = get_sents('./data/train.txt')
print(len(sentences))