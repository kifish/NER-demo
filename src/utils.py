import re,os
from collections import defaultdict
import numpy as np
from functools import reduce
from sklearn_crfsuite import metrics
from seqeval.metrics import classification_report
import torch


def load_data_and_labels(path):
    print('reading data from {} ...'.format(path))
    split_pattern = re.compile(r'[；;。，、？！\.\?,! ]')
    x = []
    y = []
    sentence = []
    labels = []
    seq_max_len = 99 # 有些句子非常长,设定最长句子为100
    with open(path,'r',encoding = 'utf8') as f:
        for line in f.readlines(): # 每行为一个字符和其tag，中间用tab隔开
            line = line.strip().split('\t')
            if(not line or len(line) < 2): continue
            word, tag = line[0], line[1]
            tag = tag if tag != 'OO' else 'O'
            if split_pattern.match(word) or len(sentence) == seq_max_len:
                sentence.append(word)
                labels.append(tag)
                x.append(sentence.copy())
                y.append(labels.copy())
                sentence.clear()
                labels.clear()
            else:
                sentence.append(word)
                labels.append(tag)
        if(len(sentence)):
            x.append(sentence.copy())
            sentence.clear()
            y.append(labels.copy())
            labels.clear()
    print('done .')
    return x,y


def load_data(path):
    print('reading data from {} ...'.format(path))
    split_pattern = re.compile(r'[；;。，、？！\.\?,! ]')
    x = []
    raw = []
    seq_max_len = 99 # 有些句子非常长, 设定最长句子为100
    with open(path,'r',encoding = 'utf8') as f: # 这里是按行来分割的。
        for line in f.readlines():
            line = line.strip()
            raw.append(line)
            seqs = []
            seq = []
            for word in line:
                if split_pattern.match(word) or len(seq) == seq_max_len:
                    seq.append(word)
                    seqs.append(seq)
                    seq = []
                else:
                    seq.append(word)
            if seq:
                seqs.append(seq)
                
            x.append(seqs)
    
    # print(x) 
    x = [seq for line in x for seq in line] # 简单按句子来分割，一个句子即一个样本
    
    # print(x)
    # with open(os.path.join(os.path.dirname(path),'processed_test.txt'),'w',encoding='utf8') as f:
    #     for line in raw:
    #         for ch in line:
    #             f.write(ch + '\n')
    #         f.write('\n')
    
    print('done .')
    
    return x



# 中文NER; 目前不支持英文NER, 因为英文NER需要考虑tokenization
class TagEncoder():
    def __init__(self, max_len = 100):
        self.max_seq_len = max_len # 不包括CLS和SEP
        self.tags  = ['B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'O'] # 7
        self.tag2id = {tag : idx + 1 for idx,tag in enumerate(self.tags)}
        self.id2tag = {idx + 1 : tag for idx,tag in enumerate(self.tags)}
        self.tag2id['padding'] = 0
        # 要加padding tag，因为x输入的时候做了padding，所以y也需要做padding
        self.id2tag[0] = 'padding'
        
    def to_ids(self,Y, postprocess = False):
        # Y: 2-d list
        res = []
        for y in Y:
            seq = []
            for tag in y: # y:seq_label
                seq.append(self.tag2id.get(tag, 7))
            seq_len = len(seq)
            
            if postprocess:
                if seq_len > self.max_seq_len:
                    seq = seq[:self.max_seq_len]
                elif seq_len < self.max_seq_len:
                    seq = seq + [0] * (self.max_seq_len - seq_len)
                    
            res.append(seq)
            
        # res: 2-d list
        return res
    
    def single_to_ids(self, y, postprocess = False):
        # Y: 1-d list
        seq = []
        for tag in y: 
            seq.append(self.tag2id.get(tag, 7))
        
        if postprocess:
            seq_len = len(seq)
            if seq_len > self.max_seq_len:
                seq = seq[:self.max_seq_len]
            
            elif seq_len < self.max_seq_len:
                seq = seq + [0] * (self.max_seq_len - seq_len) # bert是绝对位置编码
        
        return seq
    
    
    def to_tag(self, Y, logger = None):
        # Y: 2-d list
        res = []
        for y in Y:
            seq_tag = []
            for tag_id in y:
                # 有可能预测为padding
                # tag = self.id2tag[tag_id]
                tag = self.id2tag.get(tag_id, 'unk') # start or end tag
                if tag == 'unk':
                    if logger:
                        logger.info('WARNING! found unk tag') # <START> or <STOP>
                        logger.info('WARNING! tag id : {}'.format(tag_id))
                    else:
                        print('WARNING! found unk tag') # <START> or <STOP>
                        print('WARNING! tag id : {}'.format(tag_id)) 
                    tag = 'O'
                seq_tag.append(tag)
            res.append(seq_tag)
        # 2-d list
        return res
    
    

def load_processed_data(path = '../data/processed_test.txt'):
    x = []
    with open(path,'r',encoding='utf8') as f:
        for line in f.readlines():
            x.append(line.strip())
    return x

def save_pred(pred,save_path = '../data/val_pred.txt',for_validation = True):
    if for_validation:
        with open(save_path, 'w', encoding='utf8') as f:
            x_val, _ = load_data_and_labels('../data/dev.txt')
            x_val = reduce(lambda x,y: x + y,x_val)
            pred = reduce(lambda x,y: x + y,pred) # 2维list
            result = zip(x_val, pred)
            for item in result:
                f.write(item[0] + '\t' + item[1] + '\n')
    else: # save pred for test data
        with open(save_path, 'w', encoding='utf8') as f:
            x_test = load_processed_data()
            pred = reduce(lambda x,y: x + y,pred) # 2维list/tuple
            idx2 = 0
            for idx1 in range(len(x_test)):
                if x_test[idx1] == '':
                    f.write('\n')
                else:
                    f.write(x_test[idx1] + '\t' + pred[idx2] + '\n') # 为了保留空行
                    idx2 += 1


# https://stackoverflow.com/questions/9452775/converting-numpy-dtypes-to-native-python-types     
# recursion
def convert(d):
    for k,v in d.items():
        if isinstance(v, dict):
            convert(v)
        elif isinstance(v, np.generic):
            d[k] = v.item()


def eval(y_test, pred , y_test_ids = None, pred_ids = None, tag_encoder = None):
    # y_test: 2-d list; pred: 2-d list
    
    # print('-'*30) 
    # print(y_test[:10])
    # print(pred[:10])
    
    # 去掉padding
    new_y_test = []
    new_pred = []
    single_pred = []
    single_test = []
    for i, seq in enumerate(y_test):
        for j, tag in enumerate(seq):
            if j == 0: # 第一个位置是[CLS]对应的padding
                continue
            if tag == 'padding':
                break
            else:
                single_pred.append(pred[i][j])
                single_test.append(tag)
                
        new_y_test.append(single_test)
        new_pred.append(single_pred)
        single_pred = []
        single_test = []
        
    y_test = new_y_test
    pred = new_pred
    
    # print('-'*30) 
    # print(y_test)
    # print(pred)
    
    result1 = classification_report(y_test, pred, digits = 4, output_dict = True, suffix=False)
    result1_str = classification_report(y_test, pred, digits = 4, output_dict = False, suffix=False)
    
    # print(result1_str)
    
    result1['entity_macro_avg'] = result1.pop('macro avg')
    result1['entity_micro_avg'] = result1.pop('micro avg')
    result1['entity_weighted_avg'] = result1.pop('weighted avg')

    
    if y_test_ids is None and tag_encoder is None:
        tag_encoder = TagEncoder()
        
    target_names = ['B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']
    inclued_labels = list(range(1, 7)) # 1-6
    
    if y_test_ids is None:
        y_test_ids = tag_encoder.to_ids(y_test) #  [[int]]
    if pred_ids is None:
        pred_ids = tag_encoder.to_ids(pred) #  [[int]]
    
    
    # 2年后, 接口变了...
    result2 = metrics.flat_classification_report(
        y_test_ids, pred_ids, labels = inclued_labels, target_names = target_names, digits=4, output_dict = True
    )
    
    
    result2_str = metrics.flat_classification_report(
        y_test_ids, pred_ids, labels = inclued_labels, target_names = target_names, digits=4, output_dict = False
    )
    
    
    result2['token_macro_avg'] = result2.pop('macro avg')
    result2['token_micro_avg'] = result2.pop('micro avg')
    result2['token_weighted_avg'] = result2.pop('weighted avg')
    
    # print(result2_str)
    
    result1.update(result2)
    convert(result1)
    return result1, result1_str, result2_str


def cal_elapsed_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = float(format((elapsed_time - (elapsed_mins * 60)), '.2f'))
    return elapsed_mins, elapsed_secs



def batchfy_wrapper(device, with_tag = True):
    def batchfy(batch_samples):
        b_input_ids = []        
        b_target_ids = []
        for sample in batch_samples:
            b_input_ids.append(sample['input_ids'])
            if with_tag:
                b_target_ids.append(sample['target_ids'])

        b_input_ids = torch.tensor(b_input_ids, device = device, dtype=torch.long)
        
        b_attention_mask = torch.ne(b_input_ids, 0).float()
        b_attention_mask = b_attention_mask.to(device)
        
        if with_tag:
            b_target_ids = torch.tensor(b_target_ids, device = device, dtype=torch.long)
            
            return b_input_ids, b_attention_mask, b_target_ids 
        else:
            return b_input_ids, b_attention_mask
    
    return batchfy



if __name__ == '__main__':
    
    # y_true = [['O', 'O', 'O', 'B-PER', 'O'], ['B-PER', 'I-PER', 'O']]
    # y_pred = [['O', 'O', 'O', 'B-PER', 'O'], ['B-PER', 'I-PER', 'O']]

    y_true = [['O', 'O', 'O', 'B-PER', 'O','padding'], ['B-PER', 'I-PER', 'O','padding']]
    y_pred = [['O', 'O', 'O', 'B-PER', 'O','padding'], ['B-PER', 'I-PER', 'O','padding']]
    
    result, result1_str, result2_str = eval(y_test = y_true, pred = y_pred)
    
    print(result1_str)

    print(result2_str)
    
    import pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(result)
    
    # without 'convert'
    print(type(result['token_weighted_avg']['support'])) # <class 'int'>
    print(type(result['PER']['support'])) # <class 'numpy.int64'>
    print(type(result['B-PER']['support'])) # <class 'int'>
    print(type(result['B-PER']['recall'])) # <class 'float'>
    # --------------------
    
    # test_x = load_data('data/test.txt')
    # print(len(test_x)) # 29711
    # print(len(test_x[0])) # 15
    # print(len(test_x[0][0])) # 1
    # print(test_x[0])
    # print(test_x[:10])
    
    # --------------------
    
    
    
    
    