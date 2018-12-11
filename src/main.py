from src.model import BiLSTM_crf
from src.utils import load_data_and_labels,save_pred,transformer_x,transformer_y
from sklearn_crfsuite import metrics


x_train, y_train = load_data_and_labels('../data/train.txt')
transformer_x = transformer_x()
x_train = transformer_x.fit(x_train)
vocab_size = transformer_x.vocab_size
transformer_y = transformer_y(transformer_x.max_len)
y_train = transformer_y.to_onehot(y_train)

model = BiLSTM_crf(num_labels=8, word_vocab_size = vocab_size,max_seq_len=transformer_x.max_len)
model = model.build()
model.summary()


# model.fit(x_train,y_train,verbose = 1,batch_size = 12,epochs= 1)
model.fit(x_train[:1000],y_train[:1000],verbose = 1,epochs= 1)

x_test, y_test = load_data_and_labels('../data/dev.txt')
x_test = transformer_x.tran(x_test)
pred = model.predict(x_test)
pred = transformer_y.to_tag(pred)
save_pred(pred)

print(metrics.flat_f1_score(y_test, pred,
                      average='weighted', labels=transformer_y.tags))

