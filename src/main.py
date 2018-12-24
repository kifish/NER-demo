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

model = BiLSTM(num_labels = 8,max_seq_len = transformer_x.max_len,word_vocab_size = vocab_size)
model = model.build()
model.summary()

x_val, y_val = load_data_and_labels('../data/dev.txt')
x_val = transformer_x.tran(x_val)
y_val_onehot = transformer_y.to_onehot(y_val)
model.fit(x_train,y_train,validation_data = (x_val,y_val_onehot),verbose = 1,epochs= 15)




decoder = Viterbi()
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







