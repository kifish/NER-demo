from logging import error
from numpy.lib.npyio import fromregex
import torch.nn as nn 
from transformers import BertModel, BertConfig, BertPreTrainedModel
from utils import TagEncoder
import torch

'''
https://github.com/lonePatient/BERT-NER-Pytorch/blob/master/models/bert_for_ner.py#L12

https://github.com/kamalkraj/BERT-NER/blob/dev/bert.py#L15

https://huggingface.co/transformers/_modules/transformers/modeling_bert.html
'''

class BertSoftmaxForNER(BertPreTrainedModel):
    def __init__(self, model_config, **kwargs):
        bert_config = BertConfig.from_pretrained(model_config['base_model_name'], cache_dir = model_config['cache_dir'])
        super(BertSoftmaxForNER, self).__init__(bert_config)
        
        self.num_labels = model_config['num_labels']
        self.bert = BertModel.from_pretrained(model_config['base_model_name'], cache_dir = model_config['cache_dir'])
        self.dropout = nn.Dropout(model_config['hidden_dropout_prob'])
        self.classifier = nn.Linear(model_config['hidden_size'], model_config['num_labels']) # 包括padding
        
    def forward(self, input_ids, attention_mask=None,
            head_mask=None, labels=None):
        outputs = self.bert(input_ids = input_ids, attention_mask=attention_mask) 
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output) # bsz, seq_len, hidden_dim
        logits = self.classifier(sequence_output) # bsz, seq_len, num_labels
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        return outputs  #scores, (hidden_states), (attentions)      
        

# https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#bi-lstm-conditional-random-field-discussion


# crf on gpu; no batch ; very slow
# class BertCRFForNER(BertPreTrainedModel):
#     def __init__(self, model_config, **kwargs):
#         bert_config = BertConfig.from_pretrained(model_config['base_model_name'], cache_dir = model_config['cache_dir'])
#         super(BertCRFForNER, self).__init__(bert_config)
        
#         self.hidden_dim = model_config['hidden_size'] # 768
        
#         tags = ['B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'O'] # 7
#         tag_to_ix = {tag : idx + 1 for idx, tag in enumerate(tags)}
#         tag_to_ix['padding'] = 0
#         self.START_TAG = '<START>'
#         self.STOP_TAG = '<STOP>'
#         tag_to_ix[self.START_TAG] = 8
#         tag_to_ix[self.STOP_TAG] = 9
#         self.tag_to_ix = tag_to_ix
#         self.tagset_size = len(tag_to_ix) # 8 + 2 = 10
#         use_cuda = model_config['use_cuda']
#         # use_cuda = False
#         # print(use_cuda)
#         # print(type(use_cuda))
        
#         # self.device = torch.device('cuda' if use_cuda else 'cpu')
#         # if use_cuda:
#         #     self.device = torch.device('cuda')
#         # else:
#         #     self.device = torch.device('cpu')
#         self.device_name = 'cuda' if use_cuda else 'cpu'

#         self.bert = BertModel.from_pretrained(model_config['base_model_name'], cache_dir = model_config['cache_dir'])
#         self.dropout = nn.Dropout(model_config['hidden_dropout_prob'])
        
#         # Maps the output of the LSTM into tag space.
#         self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

#         # Matrix of transition parameters.  Entry i,j is the score of
#         # transitioning *to* i *from* j. ; j->i
#         self.transitions = nn.Parameter(
#             torch.randn(self.tagset_size, self.tagset_size))

#         # These two statements enforce the constraint that we never transfer
#         # to the start tag and we never transfer from the stop tag

#         self.transitions.data[tag_to_ix[self.START_TAG], :] = -10000
#         self.transitions.data[:, tag_to_ix[self.STOP_TAG]] = -10000
     

#     @staticmethod
#     def argmax(vec):
#         # return the argmax as a python int
#         # vec: [1, target_size]
#         _, idx = torch.max(vec, 1)
#         return idx.item()


#     @staticmethod
#     # Compute log sum exp in a numerically stable way for the forward algorithm
#     def log_sum_exp(vec):
#         # 单个样本:[1, target_size]
#         max_score = vec[0, BertCRFForNER.argmax(vec)] # scalar
#         max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1]) # scalar -> [1, target_size]
#         return max_score + \
#             torch.log(torch.sum(torch.exp(vec - max_score_broadcast))) 
        
#         # log(sum(exp(vec)))的稳定版
    
#     # crf
#     def _single_forward_alg(self, feats):
#         # feats: [seq_len, tagset_size]
#         # Do the forward algorithm to compute the partition function
#         init_alphas = torch.full((1, self.tagset_size), -10000., device = torch.device(self.device_name))
#         # START_TAG has all of the score.
#         init_alphas[0][self.tag_to_ix[self.START_TAG]] = 0.

#         # Wrap in a variable so that we will get automatic backprop
#         forward_var = init_alphas
#         # forward_var: [1, tagset_size], forward_var[0][tag]: 到当前的时间步的以某tag结尾的分数score;forward_var只存一个时间步
#         # Iterate through the sentence
#         # feats:[seq_len, tagset_size]
#         for feat in feats:
#             # feat: [tagset_size]
#             # feat[next_tag]: next_tag的发射分数
#             alphas_t = []  # The forward tensors at this timestep
            
#             for next_tag in range(self.tagset_size): # 遍历可能的tag
#                 # broadcast the emission score: it is the same regardless of
#                 # the previous tag
#                 emit_score = feat[next_tag].view(
#                     1, -1).expand(1, self.tagset_size)
#                 # scalar -> [1, tagset_size]
                
#                 # the ith entry of trans_score is the score of transitioning to
#                 # next_tag from i
#                 trans_score = self.transitions[next_tag].view(1, -1)
#                 # x to next_tag的转移分数: [1, tagset_size]; 有可能的tagset_size个tag转移到next_tag
                
#                 # The ith entry of next_tag_var is the value for the
#                 # edge (i -> next_tag) before we do log-sum-exp
#                 next_tag_var = forward_var + trans_score + emit_score
#                 # The forward variable for this tag is log-sum-exp of all the
#                 # scores.
#                 alphas_t.append(BertCRFForNER.log_sum_exp(next_tag_var).view(1))
            
#             # alphas_t: list, len is the tagset_size
#             forward_var = torch.cat(alphas_t).view(1, -1) # [1, tagset_size]
#             # 下一个时间步的前向变量
            
#         terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]] # 最后一步只有转移分数; STOP_TAG没有发射分数
#         alpha = BertCRFForNER.log_sum_exp(terminal_var) # 所有tag序列的分数总和
#         return alpha

#     # 单个样本
#     def _single_get_bert_features(self, input_ids, attention_mask):
#         # input_ids: [1, seq_len]
#         feats = self._get_bert_features(input_ids, attention_mask)
#         sentence_feats = feats[0]
#         return sentence_feats
    
#     # batch
#     def _get_bert_features(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids = input_ids, attention_mask=attention_mask) 
#         sequence_output = outputs[0]
#         sequence_output = self.dropout(sequence_output) # bsz, seq_len, hidden_dim
#         feats = self.hidden2tag(sequence_output) # bsz, seq_len, tagset_size
#         return feats
    

#     def _single_score_sentence(self, feats, tags):
#         # Gives the score of a provided tag sequence
#         # 指定了tag
#         # tags: tensor: [seq_len],不包括START_TAG和STOP_TAG
#         # feats: tensor: [seq_len, tagset_size], 不包括START_TAG和STOP_TAG
#         score = torch.zeros(1, device = torch.device(self.device_name))
#         tags = torch.cat([torch.tensor([self.tag_to_ix[self.START_TAG]], dtype=torch.long, device = torch.device(self.device_name)), \
#                             tags])
#         # [seq_len+1]
#         for i, feat in enumerate(feats):
#             # feat: [tagset_size] 发射分数
            
#             # 转移分数; 从tags[i]转移到tags[i + 1];
#             # 第一个时间步是tags[i]是START_TAG;
#             score = score + \
#                     self.transitions[tags[i + 1], tags[i]] + \
#                     feat[tags[i + 1]]  # 发射分数
                    
#         score = score + self.transitions[self.tag_to_ix[self.STOP_TAG], tags[-1]] # [1]
#         return score
    
#     # https://github.com/pytorch/tutorials/blob/master/beginner_source/nlp/advanced_tutorial.py
#     # https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#exercise-a-new-loss-function-for-discriminative-tagging
    
#     def _single_viterbi_decode(self, feats):
#         # feats: [seq_len, target_size]
#         # 找到分数最高的tag序列
#         backpointers = [] # 2-d list

#         # Initialize the viterbi variables in log space
#         init_vvars = torch.full((1, self.tagset_size), -10000., device = torch.device(self.device_name)) # [1, tagset_size] # 包括START_TAG和END_TAG
#         init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0

#         # forward_var at step i holds the viterbi variables for step i-1
#         forward_var = init_vvars
#         for feat in feats:
#             # 遍历每个时间步
#             # feat tagset_size
#             bptrs_t = []  # holds the backpointers for this step; 
#             # 1-d list
#             # bptrs_t[i]表示转移到tag i的是哪一个tag
#             viterbivars_t = []  # holds the viterbi variables for this step
#             # 1-d list;list of tensor; [tagset_size]; viterbivars_t[i]:以tag i结尾的最大分数
#             for next_tag in range(self.tagset_size):
#                 # next_tag_var[i] holds the viterbi variable for tag i at the
#                 # previous step, plus the score of transitioning
#                 # from tag i to next_tag.
#                 # We don't include the emission scores here because the max
#                 # does not depend on them (we add them in below)
#                 next_tag_var = forward_var + self.transitions[next_tag] # [1, tagset_size]
#                 best_tag_id = BertCRFForNER.argmax(next_tag_var) # python int;从哪个tag转移到当前tag的
#                 bptrs_t.append(best_tag_id)
#                 viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                
#             # Now add in the emission scores, and assign forward_var to the set
#             # of viterbi variables we just computed
#             forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1) # 加上下一个时间步的发射分数
#             backpointers.append(bptrs_t)

#         # Transition to STOP_TAG
#         terminal_var = forward_var + self.transitions[self.tag_to_ix[self.STOP_TAG]]
#         best_tag_id = BertCRFForNER.argmax(terminal_var) # 这里面没有STOP_TAG
#         path_score = terminal_var[0][best_tag_id]

#         # Follow the back pointers to decode the best path.
#         best_path = [best_tag_id]
#         for bptrs_t in reversed(backpointers):
#             best_tag_id = bptrs_t[best_tag_id]
#             best_path.append(best_tag_id)
#         # Pop off the start tag (we dont want to return that to the caller)
#         start = best_path.pop()
#         assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
#         best_path.reverse()
#         # path_score: tensor scalar
#         return path_score, best_path # [int]      

    
#     def single_neg_log_likelihood(self, sentence, attention_mask, tags):
#         # sentence是单个句子; tensor ids;[1,seq_len]; # 不包括START_TAG和STOP_TAG
#         # tags是单个句子的tag序列
#         sentence_feats = self._single_get_bert_features(input_ids = sentence, attention_mask = attention_mask) 
#         # feats: seq_len, tagset_size
#         forward_score = self._single_forward_alg(sentence_feats) # 前向分数; 不包括START_TAG和STOP_TAG
#         gold_score = self._single_score_sentence(sentence_feats, tags) # 指定了tag; 不包括START_TAG和STOP_TAG
#         loss = forward_score - gold_score # 让forward_score和gold_score更接近
#         return loss
    
#     def batch_neg_log_likelihood(self, input_ids, attention_mask, b_tags):
#         # batch
#         feats = self._get_bert_features(input_ids = input_ids, attention_mask = attention_mask) 
#         # feats: bsz, seq_len, tagset_size
#         b_loss = 0
#         bsz = feats.size(0)
#         for i in range(bsz):
#             sentence_feats = feats[i] # seq_len, tagset_size
#             forward_score = self._single_forward_alg(sentence_feats) 
#             gold_score = self._single_score_sentence(sentence_feats, b_tags[i]) 
#             loss = forward_score - gold_score # 这样设计loss可以让crf的参数更快收敛
#             b_loss = b_loss + loss
            
#         return b_loss
    
    
#     # 单条样本
#     def single_forward(self, input_ids, attention_mask = None): 
#         # input_ids: [1, seq_len]
#         # labels 不包括 start end tag 
#         # dont confuse this with _forward_alg above.
#         # Get the emission scores from the bert
#         sentence_feats = self._single_get_bert_features(input_ids = input_ids, attention_mask = attention_mask)
#         # Find the best path, given the features.
#         score, tag_seq = self._single_viterbi_decode(sentence_feats)
#         # score: tensor scalar
#         # tag_seq: tag ids; 预测的tag序列
#         return score, tag_seq
    
#     def batch_forward(self, input_ids, attention_mask = None): 
#         # input_ids: [bsz, seq_len]
#         # labels 不包括 start end tag 
#         # dont confuse this with _forward_alg above.
#         # Get the emission scores from the bert
#         feats = self._get_bert_features(input_ids = input_ids, attention_mask = attention_mask)
#         # [bsz, seq, tagset_size]
#         bsz = feats.size(0)
#         b_score = []
#         b_tag_seq = []
#         for i in range(bsz):
#             sentence_feats = feats[i]
#             # Find the best path, given the features.
#             score, tag_seq = self._single_viterbi_decode(sentence_feats)
#             # score: tensor scalar
#             # tag_seq: tag ids; 预测的tag序列
#             b_score.append(score)
#             b_tag_seq.append(tag_seq)
        
#         return b_score, b_tag_seq   
    
    
    
    
# # crf on cpu; no batch ; a little fast
# class BertCRFForNER(BertPreTrainedModel):
#     def __init__(self, model_config, **kwargs):
#         bert_config = BertConfig.from_pretrained(model_config['base_model_name'], cache_dir = model_config['cache_dir'])
#         super(BertCRFForNER, self).__init__(bert_config)
        
#         self.hidden_dim = model_config['hidden_size'] # 768
        
#         tags = ['B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'O'] # 7
#         tag_to_ix = {tag : idx + 1 for idx, tag in enumerate(tags)}
#         tag_to_ix['padding'] = 0
#         self.START_TAG = '<START>'
#         self.STOP_TAG = '<STOP>'
#         tag_to_ix[self.START_TAG] = 8
#         tag_to_ix[self.STOP_TAG] = 9
#         self.tag_to_ix = tag_to_ix
#         self.tagset_size = len(tag_to_ix) # 8 + 2 = 10
#         # use_cuda = model_config['use_cuda']
#         crf_use_cuda = False
#         # print(use_cuda)
#         # print(type(use_cuda))
        
#         # self.device = torch.device('cuda' if use_cuda else 'cpu')
#         # if use_cuda:
#         #     self.device = torch.device('cuda')
#         # else:
#         #     self.device = torch.device('cpu')
#         self.device_name = 'cuda' if crf_use_cuda else 'cpu'

#         self.bert = BertModel.from_pretrained(model_config['base_model_name'], cache_dir = model_config['cache_dir'])
#         self.dropout = nn.Dropout(model_config['hidden_dropout_prob'])
        
#         # Maps the output of the LSTM into tag space.
#         self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)

#         # Matrix of transition parameters.  Entry i,j is the score of
#         # transitioning *to* i *from* j. ; j->i
#         self.transitions = nn.Parameter(
#             torch.randn(self.tagset_size, self.tagset_size))
#         # https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html

#         # These two statements enforce the constraint that we never transfer
#         # to the start tag and we never transfer from the stop tag

#         self.transitions.data[tag_to_ix[self.START_TAG], :] = -10000
#         self.transitions.data[:, tag_to_ix[self.STOP_TAG]] = -10000


#     def transition_score(self, to_tag, from_tag):
#         return self.transitions[to_tag][from_tag].to(self.device_name)
    
#     def transition_scores(self, to_tag):
#         return self.transitions[to_tag].to(self.device_name)

#     @staticmethod
#     def argmax(vec):
#         # return the argmax as a python int
#         # vec: [1, target_size]
#         _, idx = torch.max(vec, 1)
#         return idx.item()


#     @staticmethod
#     # Compute log sum exp in a numerically stable way for the forward algorithm
#     def log_sum_exp(vec):
#         # 单个样本:[1, target_size]
#         max_score = vec[0, BertCRFForNER.argmax(vec)] # scalar
#         max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1]) # scalar -> [1, target_size]
#         return max_score + \
#             torch.log(torch.sum(torch.exp(vec - max_score_broadcast))) 
        
#         # log(sum(exp(vec)))的稳定版
    
#     # crf
#     def _single_forward_alg(self, feats):
#         # feats: [seq_len, tagset_size]
#         # Do the forward algorithm to compute the partition function
#         init_alphas = torch.full((1, self.tagset_size), -10000., device = torch.device(self.device_name))
#         # START_TAG has all of the score.
#         init_alphas[0][self.tag_to_ix[self.START_TAG]] = 0.

#         # Wrap in a variable so that we will get automatic backprop
#         forward_var = init_alphas
#         # forward_var: [1, tagset_size], forward_var[0][tag]: 到当前的时间步的以某tag结尾的分数score;forward_var只存一个时间步
#         # Iterate through the sentence
#         # feats:[seq_len, tagset_size]
#         for feat in feats:
#             # feat: [tagset_size]
#             # feat[next_tag]: next_tag的发射分数
#             alphas_t = []  # The forward tensors at this timestep
            
#             for next_tag in range(self.tagset_size): # 遍历可能的tag
#                 # broadcast the emission score: it is the same regardless of
#                 # the previous tag
#                 emit_score = feat[next_tag].view(
#                     1, -1).expand(1, self.tagset_size)
#                 # scalar -> [1, tagset_size]
                
#                 # the ith entry of trans_score is the score of transitioning to
#                 # next_tag from i
#                 trans_score = self.transition_scores(next_tag).view(1, -1)
#                 # x to next_tag的转移分数: [1, tagset_size]; 有可能的tagset_size个tag转移到next_tag
                
#                 # The ith entry of next_tag_var is the value for the
#                 # edge (i -> next_tag) before we do log-sum-exp
#                 next_tag_var = forward_var + trans_score + emit_score
#                 # The forward variable for this tag is log-sum-exp of all the
#                 # scores.
#                 alphas_t.append(BertCRFForNER.log_sum_exp(next_tag_var).view(1))
            
#             # alphas_t: list, len is the tagset_size
#             forward_var = torch.cat(alphas_t).view(1, -1) # [1, tagset_size]
#             # 下一个时间步的前向变量
            
#         terminal_var = forward_var + self.transition_scores(self.tag_to_ix[self.STOP_TAG]) # 最后一步只有转移分数; STOP_TAG没有发射分数
#         alpha = BertCRFForNER.log_sum_exp(terminal_var) # 所有tag序列的分数总和
#         return alpha

#     # 单个样本
#     def _single_get_bert_features(self, input_ids, attention_mask):
#         # input_ids: [1, seq_len]
#         feats = self._get_bert_features(input_ids, attention_mask)
#         sentence_feats = feats[0]
#         return sentence_feats
    
#     # batch
#     def _get_bert_features(self, input_ids, attention_mask):
#         outputs = self.bert(input_ids = input_ids, attention_mask=attention_mask) 
#         sequence_output = outputs[0]
#         sequence_output = self.dropout(sequence_output) # bsz, seq_len, hidden_dim
#         feats = self.hidden2tag(sequence_output) # bsz, seq_len, tagset_size
#         # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
#         # feats = feats.to(self.device_name) # bert放到gpu上, crf放到cpu上会不会快点
#         feats = feats.to('cpu')
        
#         return feats
    

#     def _single_score_sentence(self, feats, tags):
#         # Gives the score of a provided tag sequence
#         # 指定了tag
#         # tags: tensor: [seq_len],不包括START_TAG和STOP_TAG
#         # feats: tensor: [seq_len, tagset_size], 不包括START_TAG和STOP_TAG
#         score = torch.zeros(1, device = torch.device(self.device_name))
#         if self.device_name == 'cpu':
#             tags = tags.cpu()
#         tags = torch.cat([torch.tensor([self.tag_to_ix[self.START_TAG]], dtype=torch.long, device = torch.device(self.device_name)), \
#                             tags])
#         # [seq_len+1]
#         for i, feat in enumerate(feats):
#             # feat: [tagset_size] 发射分数
            
#             # 转移分数; 从tags[i]转移到tags[i + 1];
#             # 第一个时间步是tags[i]是START_TAG;
#             score = score + \
#                     self.transition_score(tags[i + 1], tags[i]) + \
#                     feat[tags[i + 1]]  # 发射分数
                    
#         score = score + self.transition_score(self.tag_to_ix[self.STOP_TAG], tags[-1]) # [1]
#         return score
    
#     # https://github.com/pytorch/tutorials/blob/master/beginner_source/nlp/advanced_tutorial.py
#     # https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html#exercise-a-new-loss-function-for-discriminative-tagging
    
#     def _single_viterbi_decode(self, feats):
#         # feats: [seq_len, target_size]
#         # 找到分数最高的tag序列
#         backpointers = [] # 2-d list

#         # Initialize the viterbi variables in log space
#         init_vvars = torch.full((1, self.tagset_size), -10000., device = torch.device(self.device_name)) # [1, tagset_size] # 包括START_TAG和END_TAG
#         init_vvars[0][self.tag_to_ix[self.START_TAG]] = 0

#         # forward_var at step i holds the viterbi variables for step i-1
#         forward_var = init_vvars
#         for feat in feats:
#             # 遍历每个时间步
#             # feat tagset_size
#             bptrs_t = []  # holds the backpointers for this step; 
#             # 1-d list
#             # bptrs_t[i]表示转移到tag i的是哪一个tag
#             viterbivars_t = []  # holds the viterbi variables for this step
#             # 1-d list;list of tensor; [tagset_size]; viterbivars_t[i]:以tag i结尾的最大分数
#             for next_tag in range(self.tagset_size):
#                 # next_tag_var[i] holds the viterbi variable for tag i at the
#                 # previous step, plus the score of transitioning
#                 # from tag i to next_tag.
#                 # We don't include the emission scores here because the max
#                 # does not depend on them (we add them in below)
#                 next_tag_var = forward_var + self.transition_scores(next_tag) # [1, tagset_size]
#                 best_tag_id = BertCRFForNER.argmax(next_tag_var) # python int;从哪个tag转移到当前tag的
#                 bptrs_t.append(best_tag_id)
#                 viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                
#             # Now add in the emission scores, and assign forward_var to the set
#             # of viterbi variables we just computed
#             forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1) # 加上下一个时间步的发射分数
#             backpointers.append(bptrs_t)

#         # Transition to STOP_TAG
#         terminal_var = forward_var + self.transition_scores(self.tag_to_ix[self.STOP_TAG])
#         best_tag_id = BertCRFForNER.argmax(terminal_var) # 这里面没有STOP_TAG
#         path_score = terminal_var[0][best_tag_id]

#         # Follow the back pointers to decode the best path.
#         best_path = [best_tag_id]
#         for bptrs_t in reversed(backpointers):
#             best_tag_id = bptrs_t[best_tag_id]
#             best_path.append(best_tag_id)
#         # Pop off the start tag (we dont want to return that to the caller)
#         start = best_path.pop()
#         assert start == self.tag_to_ix[self.START_TAG]  # Sanity check
#         best_path.reverse()
#         # path_score: tensor scalar
#         return path_score, best_path # [int]      

    
#     def single_neg_log_likelihood(self, sentence, attention_mask, tags):
#         # sentence是单个句子; tensor ids;[1,seq_len]; # 不包括START_TAG和STOP_TAG
#         # tags是单个句子的tag序列
#         sentence_feats = self._single_get_bert_features(input_ids = sentence, attention_mask = attention_mask) 
#         # feats: seq_len, tagset_size
#         forward_score = self._single_forward_alg(sentence_feats) # 前向分数; 不包括START_TAG和STOP_TAG
#         gold_score = self._single_score_sentence(sentence_feats, tags) # 指定了tag; 不包括START_TAG和STOP_TAG
#         loss = forward_score - gold_score # 让forward_score和gold_score更接近
#         return loss
    
#     def batch_neg_log_likelihood(self, input_ids, attention_mask, b_tags):
#         # batch
#         feats = self._get_bert_features(input_ids = input_ids, attention_mask = attention_mask) 
#         # feats: bsz, seq_len, tagset_size
#         b_loss = 0
#         bsz = feats.size(0)
#         for i in range(bsz):
#             sentence_feats = feats[i] # seq_len, tagset_size
#             forward_score = self._single_forward_alg(sentence_feats) 
#             gold_score = self._single_score_sentence(sentence_feats, b_tags[i]) 
#             loss = forward_score - gold_score # 这样设计loss可以让crf的参数更快收敛
#             b_loss = b_loss + loss
            
#         return b_loss
    
    
#     # 单条样本
#     def single_forward(self, input_ids, attention_mask = None): 
#         # input_ids: [1, seq_len]
#         # labels 不包括 start end tag 
#         # dont confuse this with _forward_alg above.
#         # Get the emission scores from the bert
#         sentence_feats = self._single_get_bert_features(input_ids = input_ids, attention_mask = attention_mask)
#         # Find the best path, given the features.
#         score, tag_seq = self._single_viterbi_decode(sentence_feats)
#         # score: tensor scalar
#         # tag_seq: tag ids; 预测的tag序列
#         return score, tag_seq
    
#     def batch_forward(self, input_ids, attention_mask = None): 
#         # input_ids: [bsz, seq_len]
#         # labels 不包括 start end tag 
#         # dont confuse this with _forward_alg above.
#         # Get the emission scores from the bert
#         feats = self._get_bert_features(input_ids = input_ids, attention_mask = attention_mask)
#         # [bsz, seq, tagset_size]
#         bsz = feats.size(0)
#         b_score = []
#         b_tag_seq = []
#         for i in range(bsz):
#             sentence_feats = feats[i]
#             # Find the best path, given the features.
#             score, tag_seq = self._single_viterbi_decode(sentence_feats)
#             # score: tensor scalar
#             # tag_seq: tag ids; 预测的tag序列
#             b_score.append(score)
#             b_tag_seq.append(tag_seq)
        
#         return b_score, b_tag_seq   
    
    
# use crf lib
# support gpu
# batch
# very fast
from torchcrf import CRF
class BertCRFForNER(BertPreTrainedModel):
    def __init__(self, model_config, **kwargs):
        bert_config = BertConfig.from_pretrained(model_config['base_model_name'], cache_dir = model_config['cache_dir'])
        super(BertCRFForNER, self).__init__(bert_config)
        
        self.hidden_dim = model_config['hidden_size'] # 768
        
        tags = ['B-LOC', 'I-LOC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER', 'O'] # 7
        tag_to_ix = {tag : idx + 1 for idx, tag in enumerate(tags)}
        tag_to_ix['padding'] = 0
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix) # 8
        
        self.bert = BertModel.from_pretrained(model_config['base_model_name'], cache_dir = model_config['cache_dir'])
        self.dropout = nn.Dropout(model_config['hidden_dropout_prob'])
        
        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)
        
        self.crf = CRF(self.tagset_size, batch_first = True)
        # https://pytorch-crf.readthedocs.io/en/stable/#torchcrf.CRF.forward
        
        
    # batch
    def _get_bert_features(self, input_ids, attention_mask):
        outputs = self.bert(input_ids = input_ids, attention_mask=attention_mask) 
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output) # bsz, seq_len, hidden_dim
        feats = self.hidden2tag(sequence_output) # bsz, seq_len, tagset_size
        # https://pytorch.org/tutorials/beginner/blitz/data_parallel_tutorial.html
        # feats = feats.to(self.device_name) # bert放到gpu上, crf放到cpu上会不会快点
        # feats = feats.to('cpu')        
        return feats
    
    def batch_neg_log_likelihood(self, input_ids, attention_mask, b_tags):
        # batch
        feats = self._get_bert_features(input_ids = input_ids, attention_mask = attention_mask) 
        # feats: bsz, seq_len, tagset_size
        # print(attention_mask) # tensor float
        attention_mask = attention_mask.byte() # -> torch.uint8
        log_likelihood= self.crf.forward(feats, b_tags, mask= attention_mask, reduction='mean') # mean seq loss
        # all only supports torch.uint8 and torch.bool dtypes
        loss = -log_likelihood
        return loss
    
    
    def batch_forward(self, input_ids, attention_mask = None): 
        # input_ids: [bsz, seq_len]
        # labels 不包括 start end tag 
        # dont confuse this with _forward_alg above.
        # Get the emission scores from the bert
        feats = self._get_bert_features(input_ids = input_ids, attention_mask = attention_mask)
        # [bsz, seq, tagset_size]
        # print(attention_mask)
        attention_mask = attention_mask.byte() # -> uint8
        b_tag_seq = self.crf.decode(feats, mask = attention_mask)
        # tag_seq: tag ids; 预测的tag序列; List[List[int]]
        # 会把padding部分截断
        # print(b_tag_seq)
        # print(type(b_tag_seq))
        # raise EOFError
        return None, b_tag_seq   

    
 
if __name__ == '__main__':
    
    # from transformers import AutoTokenizer, AutoModel
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", cache_dir = './models/bert/')
    # model = AutoModel.from_pretrained("bert-base-chinese", cache_dir = './models/bert/')
    # text = '你好'
    # input = tokenizer.encode(text, text_pair=None, add_special_tokens=True)
    # import torch 
    # input = torch.tensor([input], dtype = torch.long)
    # print(input) # tensor([[ 101,  872, 1962,  102]])
    # output = model(input)
    # print(len(output)) # 2
    # print(output[0].size()) # torch.Size([1, 4, 768])
    
    # --------------------------------
    # from transformers import AutoTokenizer, AutoModel
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", cache_dir = './models/bert/')
    # model_config = {
    #     'base_model_name': 'bert-base-chinese',
    #     'cache_dir': './models/bert/',
    #     'hidden_size': 768,
    #     'hidden_dropout_prob': 0.1,
    #     'num_labels': 7,
    # }
    # model = BertSoftmaxForNER(model_config)
    # text = '你好'
    # input = tokenizer.encode(text, text_pair=None, add_special_tokens=True)
    # import torch 
    # input = torch.tensor([input], dtype = torch.long)
    # print(input) # tensor([[ 101,  872, 1962,  102]])
    # output = model(input)
    # print(len(output)) # 1
    # print(output[0].size()) # torch.Size([1, 4, 7])
    
    # --------------------------------

    from transformers import AutoTokenizer
    # Make up some training data
    training_data = [
        (
        "测 试 数 据".split(),
        "O O O O".split()
        ), 
        (
        "网 易 公 司".split(),
        "B-ORG I-ORG I-ORG I-ORG".split()
        )
    ]

    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", cache_dir = './models/bert/')
    b_x = []
    b_y = []
    tag_encoder = TagEncoder()
    model_config = {
        'base_model_name': 'bert-base-chinese',
        'cache_dir': './models/bert/',
        'hidden_size': 768,
        'hidden_dropout_prob': 0.1,
        'use_cuda': False
    }
    model = BertCRFForNER(model_config)
    model.eval()
    
    for x,y in training_data:
        x = tokenizer.convert_tokens_to_ids(x)
        y = tag_encoder.single_to_ids(y)
        b_x.append(x)
        b_y.append(y)
        x = torch.tensor([x], dtype=torch.long)
        print(model.single_forward(x))
        
    print('-'*30)
    print(b_x)
    print(b_y)
    print(model.batch_forward(torch.tensor(b_x, dtype=torch.long)))
    
    
    # ---------------------------------------------
    
    



