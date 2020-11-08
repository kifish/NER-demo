import torch.nn as nn 
from transformers import BertModel, BertConfig, BertPreTrainedModel


'''
中文NER; 目前不支持英文NER, 因为英文NER需要考虑tokenization
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
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", cache_dir = './models/bert/')
    model_config = {
        'base_model_name': 'bert-base-chinese',
        'cache_dir': './models/bert/',
        'hidden_size': 768,
        'hidden_dropout_prob': 0.1,
        'num_labels': 7,
    }
    model = BertSoftmaxForNER(model_config)
    text = '你好'
    input = tokenizer.encode(text, text_pair=None, add_special_tokens=True)
    import torch 
    input = torch.tensor([input], dtype = torch.long)
    print(input) # tensor([[ 101,  872, 1962,  102]])
    output = model(input)
    print(len(output)) # 1
    print(output[0].size()) # torch.Size([1, 4, 7])
    
    


