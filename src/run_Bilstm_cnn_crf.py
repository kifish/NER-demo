import os
import numpy as np
from model import BiLSTM_cnn_crf
import utils
from utils import load_data_and_labels,load_data,save_pred,transformer_x,transformer_y

os.environ["CUDA_VISIBLE_DEVICES"]="-1" # just use cpu

# load word2vec
word2vec = {}
with open('../data/word2vec.txt','r',encoding = 'utf8') as f:
    f.readline() # 第一行是word数量以及embedding size;跳过
    for line in f.readlines():
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec

print('Found %s word vectors.' % len(word2vec))

# load data and labels
x_train, y_train = load_data_and_labels('../data/train.txt')
transformer_x = transformer_x()
x_train = transformer_x.fit(x_train)
vocab_size = transformer_x.vocab_size
transformer_y = transformer_y(transformer_x.max_len)
y_train = transformer_y.to_onehot(y_train)

#init embedding_layer
EMBEDDING_DIM = 100
embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
word2id = transformer_x.word2id
for word,vec in word2vec.items():
    if word in word2id:
        embedding_matrix[word2id[word]] = word2vec[word]

model = BiLSTM_cnn_crf(num_labels=8,embedding_matrix =embedding_matrix ,max_seq_len=transformer_x.max_len,use_crf = True)
model = model.build()
model.summary()

x_val, y_val = load_data_and_labels('../data/dev.txt')
x_val = transformer_x.tran(x_val)
y_val_onehot = transformer_y.to_onehot(y_val)
model.fit(x_train,y_train,validation_data = (x_val,y_val_onehot),verbose = 2,batch_size = 12,epochs= 15)

y_pred = model.predict(x_val)
y_pred = transformer_y.to_tag(y_pred)

utils.eval(y_val,y_pred)

x_test = load_data('../data/test.txt')


if not os.path.exists('../data/pretrain_embedding_cnn'):
    os.mkdir('../data/pretrain_embedding_cnn')
save_path = '../data/pretrain_embedding_cnn/pred.txt'
save_pred(pred,save_path)