import numpy as np
import os
from src.model import BiLSTM_cnn_crf
from src.utils import load_data_and_labels,save_pred,transformer_x,transformer_y
from sklearn_crfsuite import metrics

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

x_test, y_test = load_data_and_labels('../data/dev.txt')
x_test = transformer_x.tran(x_test)
y_test_onehot = transformer_y.to_onehot(y_test)
model.fit(x_train,y_train,validation_data = (x_test,y_test_onehot),verbose = 1,batch_size = 12,epochs= 15)

pred = model.predict(x_test)
pred = transformer_y.to_tag(pred)
print(metrics.flat_f1_score(y_test, pred,
                      average='weighted', labels=transformer_y.tags))

# group B and I results
labels = transformer_y.tags
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print(metrics.flat_classification_report(
    y_test, pred, labels=sorted_labels, digits=3
))

os.mkdir('../data/pretrain_embedding_cnn')
save_path = '../data/pretrain_embedding_cnn/pred.txt'
save_pred(pred,save_path)

