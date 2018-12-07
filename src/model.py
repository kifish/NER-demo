import re,os
import numpy as np
import pandas as pd
# python3要使用绝对路径

# backoff2005语料
s = open('../data/msr_train.txt', encoding='utf8').read()
s = s.split('\r\n')

def clean(s): #整理一下数据，有些不规范的地方
    if u'“/s' not in s:
        return s.replace(u' ”/s', '')
    elif u'”/s' not in s:
        return s.replace(u'“/s ', '')
    elif u'‘/s' not in s:
        return s.replace(u' ’/s', '')
    elif u'’/s' not in s:
        return s.replace(u'‘/s ', '')
    else:
        return s

s = u''.join(map(clean, s))
s = re.split(u'[，。！？、]/[bems]', s)

data = [] #生成训练样本
label = []
def get_xy(s):
    s = re.findall('(.)/(.)', s)
    if s:
        s = np.array(s)
        return list(s[:,0]), list(s[:,1])

for i in s:
    x = get_xy(i)
    if x:
        data.append(x[0])
        label.append(x[1])

d = pd.DataFrame(index=range(len(data)))
d['data'] = data
d['label'] = label
maxlen = 32
d = d[d['data'].apply(len) <= maxlen] # 丢掉多于32字的样本
d.index = range(len(d))


chars = [] #统计所有字，跟每个字编号
for i in data:
    chars.extend(i)

chars = pd.Series(chars).value_counts()
chars[:] = range(1, len(chars)+1)

# chars[word]可取出该词对应的索引

#生成适合模型输入的格式
from keras.utils import np_utils
d['x'] = d['data'].apply(lambda x: np.array(list(chars[x])+[0]*(maxlen-len(x)))) # padding 0
#pandas 真的很慢

tag = pd.Series({'s':0, 'b':1, 'm':2, 'e':3, 'x':4})
def trans_one(x):
    _ = map(lambda y: np_utils.to_categorical(y,5), tag[x].reshape((-1,1)))
    _ = list(_)
    _.extend([np.array([[0,0,0,0,1]])]*(maxlen-len(x)))
    return np.array(_)

# >>> [np.array([[0,0,0,0,1]])]*(2)
# [array([[0, 0, 0, 0, 1]]), array([[0, 0, 0, 0, 1]])]

d['y'] = d['label'].apply(trans_one)


#设计模型
embedding_size = 128
from keras.layers import Dense, Embedding, LSTM, TimeDistributed, Input, Bidirectional
from keras.models import Model
from keras.models import load_model
if not os.path.exists('./tmp/my_model.h5'):
    sequence = Input(shape=(maxlen,), dtype='int32')

    embedded = Embedding(len(chars)+1, embedding_size, input_length=maxlen, mask_zero=True)(sequence)
    blstm = Bidirectional(LSTM(64, return_sequences=True), merge_mode='sum')(embedded)
    output = TimeDistributed(Dense(5, activation='softmax'))(blstm)
    model = Model(input=sequence, output=output)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    batch_size = 1024
    # history = model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1,maxlen,5)), batch_size=batch_size, nb_epoch=50)
    model.fit(np.array(list(d['x'])), np.array(list(d['y'])).reshape((-1,maxlen,5)), verbose = 2,batch_size=batch_size, nb_epoch=3)

    if os.path.exists('./tmp/'):
        model.save('./tmp/my_model.h5')
    else:
        os.mkdir('./tmp/')
        model.save('./tmp/my_model.h5')

else:
    model = load_model('./tmp/my_model.h5')


#最终模型可以输出每个字属于每种标签的概率
#然后用维比特算法来dp

# >>> i = [0.2,0.2,0.3,0.3,0]
# >>> dict(zip(['s','b','m','e'], i[:4]))
# {'s': 0.2, 'b': 0.2, 'm': 0.3, 'e': 0.3}


def simple_cut(s):
    if s:
        # 遇到新词 没有索引
        # 还要注意编码的问题,索引对应的词都是utf8的
        # 切词之前要全部转换成utf8编码
        try:
            r = model.predict(np.array([list(chars[list(s)].fillna(0).astype(int))+[0]*(maxlen-len(s))]), verbose=False)[0][:len(s)]
            r = np.log(r)
            nodes = [dict(zip(['s','b','m','e'], i[:4])) for i in r]
            t = viterbi(nodes)
            words = []
            for i in range(len(s)):
                if t[i] in ['s', 'b']:
                    words.append(s[i])
                else:
                    words[-1] += s[i]
            return words
        except:
            return ['无法分词:词无索引']
    else:
        return []



# 全角转半角
def DBC2SBC(input_str):
    output_str = ""
    for uchar in input_str:
        inside_code = ord(uchar)
        if inside_code == 0x3000:   #如果是空格直接替换
            inside_code = 0x0020
        elif inside_code == 65125: # '﹥'
            inside_code = 62
        elif inside_code == 65124: # '﹤'
            inside_code = 60
        elif inside_code >= 65281 and inside_code <= 65374:
            inside_code -= 0xfee0
        output_str += chr(inside_code)
    return output_str


not_cuts = re.compile(r'([\da-zA-Z ]+)|[。，、？！\.\?,!]')

def cut_word(s):
    s = DBC2SBC(s)
    result = []
    j = 0
    for i in not_cuts.finditer(s):
        result.extend(simple_cut(s[j:i.start()]))
        result.append(s[i.start():i.end()])
        j = i.end()
    result.extend(simple_cut(s[j:]))
    return result

if __name__ == '__main__':
    import time
    print(cut_word('他来到了网易杭研大厦'))
    while True:
        s = input()
        start = time.time()
        print(cut_word(s))
        print(time.time()-start)