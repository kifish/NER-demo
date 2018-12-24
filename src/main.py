from model import BiLSTM,Viterbi
import utils
from utils import load_data_and_labels,save_pred,transformer_x,transformer_y
import numpy as np

x_train, y_train = load_data_and_labels('../data/train.txt')
transformer_x = transformer_x()
x_train = transformer_x.fit(x_train)
vocab_size = transformer_x.vocab_size
transformer_y = transformer_y(transformer_x.max_len)
y_train = transformer_y.to_onehot(y_train)

use_pretrain_embedding = True
if use_pretrain_embedding:
    # load word2vec
    word2vec = {}
    with open('../data/word2vec.txt', 'r', encoding='utf8') as f:
        f.readline()  # 第一行是word数量以及embedding size;跳过
        for line in f.readlines():
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            word2vec[word] = vec

    print('Found %s word vectors.' % len(word2vec))

    # init embedding_layer
    EMBEDDING_DIM = 100
    embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    word2id = transformer_x.word2id
    for word, vec in word2vec.items():
        if word in word2id:
            embedding_matrix[word2id[word]] = word2vec[word]

    model = BiLSTM(num_labels=8, embedding_matrix=embedding_matrix, max_seq_len=transformer_x.max_len)

else:
    model = BiLSTM(num_labels = 8,max_seq_len = transformer_x.max_len,word_vocab_size = vocab_size)

model = model.build()
model.summary()

x_val, y_val = load_data_and_labels('../data/dev.txt')
x_val = transformer_x.tran(x_val)
y_val_onehot = transformer_y.to_onehot(y_val)
model.fit(x_train,y_train,validation_data = (x_val,y_val_onehot),verbose = 1,epochs= 15)

# decode
use_offset = True
if use_offset:
    print('use tran_with_offset')
decoder = Viterbi(use_offset)
tags = ['B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'O']

y_pred_raw = model.predict(x_val)
y_pred = []
for idx, prob_distributions in enumerate(y_pred_raw):
    # 去掉padding部分
    prob_distributions = prob_distributions[-len(y_val[idx]):]
    prob_distributions = np.log(prob_distributions)
    nodes = [dict(zip(tags,prob_distribution[1:])) for prob_distribution in prob_distributions]
    the_tags = decoder.viterbi(nodes)
    y_pred.append(the_tags)

utils.eval(y_val,y_pred)
save_pred(y_pred)







