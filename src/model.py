from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, TimeDistributed
from keras.models import Model
import numpy as np

class BiLSTM(object):
    def __init__(self,
                 num_labels,
                 max_seq_len,
                 word_vocab_size=None,
                 char_vocab_size=None,
                 word_embedding_dim=100,
                 char_embedding_dim=25,
                 word_lstm_size=120,
                 char_lstm_size=25,
                 fc_dim=100,
                 dropout=0.5,
                 embedding_matrix=None,
                 use_char=False,
                 optimizer = 'adam'):
        self.char_embedding_dim = char_embedding_dim
        self.word_embedding_dim = word_embedding_dim
        self.char_lstm_size = char_lstm_size
        self.word_lstm_size = word_lstm_size
        self.char_vocab_size = char_vocab_size
        self.word_vocab_size = word_vocab_size
        self.fc_dim = fc_dim
        self.dropout = dropout
        self.use_char = use_char
        self.embedding_matrix = embedding_matrix
        self.num_labels = num_labels
        self.optimizer = optimizer
        self.max_seq_len = max_seq_len
    def build(self,metrics = ['accuracy']):
        word_id = Input(shape=(self.max_seq_len,), dtype='int32', name='word_input') # 输入是一个句子，该句子经过padding处理后是定长的 self.max_seq_len ,句子不是word的序列而是word_id的序列
        inputs = [word_id] # 之后可以append char2id;不过中文没有char2id,除非考虑偏旁
        if self.embedding_matrix is None:
            word_embedding = Embedding(input_dim = self.word_vocab_size,  \
                                       output_dim = self.word_embedding_dim, \
                                       mask_zero= True,  \
                                       input_length = self.max_seq_len,
                                       name= 'word_embedding')(word_id)
        else:
            word_embedding = Embedding(input_dim=self.embedding_matrix.shape[0], \
                                        output_dim=self.embedding_matrix.shape[1], \
                                        mask_zero=True, \
                                        weights=[self.embedding_matrix],
                                       input_length=self.max_seq_len,
                                       trainable = True,
                                        name='word_embedding')(word_id)
        dropout = Dropout(self.dropout,name = 'dropout')(word_embedding)
        z = Bidirectional(LSTM(units=self.word_lstm_size,return_sequences = True,recurrent_dropout=0.1), merge_mode='sum',name = 'BiLSTM')(dropout)
        fc = TimeDistributed(Dense(self.fc_dim, activation= 'tanh',name= 'time_distributed_fc_layer'))(z)
        loss = 'categorical_crossentropy' #交叉熵
        output = TimeDistributed(Dense(self.num_labels, activation= 'softmax'))(fc)  # 归一
        model = Model(inputs = inputs, outputs = output)
        model.compile(loss = loss,optimizer = self.optimizer,metrics = metrics)
        return model
class Viterbi(object):
    def __init__(self,use_offset = False):
        # 从语料中统计得到
        self.trans_prob = {}
        if use_offset:
            path = '../data/tran_with_offset.txt'
        else:
            path = '../data/tran.txt'
        with open(path,'r',encoding= 'utf8') as f:
            for line in f.readlines():
                tran,prob = line.strip().split('\t')
                prob = eval(prob)
                self.trans_prob[tran] = prob
        self.trans_prob = {i:np.log(self.trans_prob[i]) for i in self.trans_prob.keys()}
        self.tags = ['B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'O']

    def viterbi(self,nodes):
        #nn 最后一层softmax返回的是什么，字属于每种标签的概率？
        #常规是这样理解，但是在viterbi模型中，要理解为 P(the given word | tag i)
        paths = {(tag,) : nodes[0][tag] for tag in self.tags if tag[0] != 'I'} #起始点强行优化,这里不考虑初始概率向量
        for node_idx in range(1,len(nodes)):
            paths_ = paths.copy()
            paths = {}
            for label in nodes[node_idx].keys():
                #内循环 求以给定的label(hidden state)作为最后一个label的最优路径
                nows = {}
                for path in paths_.keys():
                    pre_node_label = path[-1]
                    if pre_node_label + '->' + label in self.trans_prob.keys():
                        # 把概率相乘等价地转化为相加;(table,)的写法更robust
                        nows[path+(label,)]= paths_[path] + self.trans_prob[path[-1] + '->' + label] + nodes[node_idx][label]
                k = np.argmax(list(nows.values()))
                paths[list(nows.keys())[k]] = list(nows.values())[k]
        return list(paths.keys())[np.argmax(list(paths.values()))]

if __name__ == '__main__':
    # model = BiLSTM(num_labels= 8, word_vocab_size= 10000, max_seq_len = 100)
    # model = model.build()
    # model.summary()

    decoder = Viterbi(True)
    nodes = [{'B-LOC':0.1,'B-PER':0.1,'B-ORG':0.1,'I-LOC':0.1,'I-PER':0.1,'I-ORG':0.1,'O':0.4},{'B-LOC':0.1,'B-PER':0.1,'B-ORG':0.1,'I-LOC':0.1,'I-PER':0.1,'I-ORG':0.1,'O':0.4}]
    print(decoder.viterbi(nodes))


