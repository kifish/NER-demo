from src.model import BiLSTM_crf
from src.utils import load_data_and_labels,save_pred,transformer_x,transformer_y
from sklearn_crfsuite import metrics

x_train, y_train = load_data_and_labels('../data/train.txt')
transformer_x = transformer_x()
x_train = transformer_x.fit(x_train)
vocab_size = transformer_x.vocab_size
transformer_y = transformer_y(transformer_x.max_len)
y_train = transformer_y.to_onehot(y_train)

use_crf = True
model = BiLSTM_crf(num_labels=8, word_vocab_size = vocab_size,max_seq_len=transformer_x.max_len,use_crf = use_crf)
model = model.build()
model.summary()


# model.fit(x_train,y_train,verbose = 1,batch_size = 12,epochs= 1)
x_test, y_test = load_data_and_labels('../data/dev.txt')
x_test = transformer_x.tran(x_test)
y_test_onehot = transformer_y.to_onehot(y_test)
model.fit(x_train,y_train,validation_data = (x_test,y_test_onehot),verbose = 1,epochs= 1)

pred = model.predict(x_test)

if use_crf:
    pass
else:
    # 直接选argmax？ 这种的肯定不如verterbi解码好。
    pass
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
save_pred(pred)

