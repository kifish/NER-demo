from keras.layers import Dense, Activation, LSTM, Bidirectional, Embedding, Input, Dropout, TimeDistributed
from keras.layers.merge import Concatenate
from keras.models import Model
from keras_contrib.layers import CRF

class BiLSTM_crf(object):
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
                 use_char=True,
                 use_crf=True,
                 optimizer = 'adam'):
        """build a Bi-LSTM-crf model.
        Args:
            word_vocab_size (int): word vocabulary size.
            char_vocab_size (int): character vocabulary size.
            num_labels (int): number of entity labels.
            word_embedding_dim (int): word embedding dimensions.
            char_embedding_dim (int): character embedding dimensions.
            word_lstm_size (int): word LSTM feature extractor output dimensions.
            char_lstm_size (int): character tagger LSTM output dimensions.
            fc_dim (int): output fully-connected layer size.
            dropout (float): dropout rate.
            embeddings (numpy array): word embedding matrix.
            use_char (boolean): add char feature.
            use_crf (boolean): use crf as last layer.
        """
        self.char_embedding_dim = char_embedding_dim
        self.word_embedding_dim = word_embedding_dim
        self.char_lstm_size = char_lstm_size
        self.word_lstm_size = word_lstm_size
        self.char_vocab_size = char_vocab_size
        self.word_vocab_size = word_vocab_size
        self.fc_dim = fc_dim
        self.dropout = dropout
        self.use_char = use_char
        self.use_crf = use_crf
        self.embedding_matrix = embedding_matrix
        self.num_labels = num_labels
        self.optimizer = optimizer
        self.max_seq_len = max_seq_len
    def build(self,metrics = ['accuracy']):
        word_id = Input(shape=(self.max_seq_len,), dtype='int32', name='word_input') # 输入是一个句子，该句子经过padding处理后是定长的 self.max_seq_len ,句子不是word的序列而是word_id的序列
        inputs = [word_id] # 之后可以append char2id;不过中文没有char2id
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
        # input_dim 词表大小; output_dim 词向量维度大小
        #在TensorFlow中 Embedding层就相当于一个二维矩阵，行数即词表大小，列数即词向量维度大小，并配合lookup
        #假设输入的句子有10个word，则输出为10*word_embedding_dim,即一个二维矩阵
        dropout = Dropout(self.dropout,name = 'dropout')(word_embedding)
        z = Bidirectional(LSTM(units=self.word_lstm_size,return_sequences = True,recurrent_dropout=0.1), merge_mode='sum',name = 'BiLSTM')(dropout)
        #一个LSTM单元 对应 一个word，一个word相当于一个word_embedding_dim大小的向量，经过一个LSTM单元，变成了一个self.word_lstm_size大小的向量
        # 双向LSTM 把同一个word的向量相加了或者拼接送入下一层
        #LSTM 既提取字的特征，也提取了字所在的序列的特征
        #一个句子 在LSTM层之后变成了二维矩阵
        fc = TimeDistributed(Dense(self.fc_dim, activation= 'tanh',name= 'time_distributed_fc_layer'))(z)
        #如果在TensorFlow里面还需要把z flat之后再进fc_layer
        ## applies fully-connected operation at every timestep
        # fc = Dense(self.fc_dim, activation='tanh', name='time_distributed_fc_layer')(z)
        # 没有TimeDistributed 没法配合crf
        #fc_layer 再接一个softmax就可以得到句子中每个词的tag的概率分布，然后对每个词取概率最大的tag/或者用viterbi取概率最大的tag序列
        # 以上的两种方法，显然是不如crf，因为crf还考虑到了句子中tag序列。
        if self.use_crf:
            crf = CRF(self.num_labels, sparse_target = False,name = 'crf')
            loss = crf.loss_function
            output = crf(fc)
            model = Model(inputs = inputs, outputs = output)
            if metrics == ['accuracy']:
                metrics = [crf.accuracy]
            # 改成crf.accuracy 之后会导致倾向于预测成O
            model.compile(loss = loss,optimizer = self.optimizer,metrics = metrics)
        else:
            loss = 'categorical_crossentropy' #交叉熵
            output = TimeDistributed(Dense(self.num_labels, activation= 'softmax'))(fc)  # 归一
            model = Model(inputs = inputs, outputs = output)
            model.compile(loss = loss,optimizer = self.optimizer,metrics = metrics)
        return model


if __name__ == '__main__':
    model = BiLSTM_crf(num_labels= 8, word_vocab_size= 20000)
    model = model.build()
    model.summary()





