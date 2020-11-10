from torch.utils.data import Dataset
from tqdm import tqdm
import time, os
from transformers import AutoTokenizer
from utils import *


from contextlib import ContextDecorator

class timer_context(ContextDecorator):
    '''Elegant Timer via ContextDecorator'''
    def __init__(self, name):
        self.name = name
    def __enter__(self):
        print('{} ...'.format(self.name))
        self.start = time.time()
    def __exit__(self, *args):
        self.end = time.time()
        self.elapse = self.end - self.start
        print("Processing time for [{}] is: {} seconds".format(self.name, self.elapse))
        


class NERDataset(Dataset):
    def __init__(self, file_path, tokenizer, block_size = 102, debug = False, debug_num_exmaple = 100, verbose = False):
        self.tokenizer = tokenizer
        self.cls_id = tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
        
        X,Y = load_data_and_labels(file_path)
        all_data = []
        tag_encoder = TagEncoder(max_len = block_size - 2) # CLS;SEP
        
        
        print('convert tokens to ids ...')        
        for x,y in zip(X, Y):
            data = {
                'input_ids': self.tokenizer.convert_tokens_to_ids(x),   # https://huggingface.co/transformers/main_classes/tokenizer.html#transformers.PreTrainedTokenizer.convert_tokens_to_ids
                'target_ids': tag_encoder.single_to_ids(y, postprocess = False),
                'seq_str': ' '.join(x),
                'tag_str': ' '.join(y)
            }
            
            assert len(data['input_ids']) == len(data['target_ids']),  data
            # 确保长度都是对齐的
            
            all_data.append(data)
            
            if debug and len(all_data) == debug_num_exmaple:
                break
                
        print('done.')    
            
        self.data = all_data # list not tensor

        self.process(block_size)
        
        print('num of example : {}'.format(len(self.data)))

        if verbose:
            print('one example from {} : '.format(file_path))
            print(self.data[5])
            

    def process(self, block_size):
        text_block_size = block_size - 2
        print('processing padding ...')
        for d in self.data:                
            d['input_ids'] = d['input_ids'][:text_block_size]
            d['input_ids'] = [self.cls_id] + d['input_ids'] + [self.sep_id]
            d['input_ids'] = d['input_ids'] + [0] * (block_size - len(d['input_ids'])) 

            d['target_ids'] = d['target_ids'][:text_block_size]
            d['target_ids'] = [0] + d['target_ids'] + [0] # CLS SEP
            d['target_ids'] = d['target_ids'] + [0] * (block_size - len(d['target_ids'])) # padding token 不需要预测

        print('done')

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)




class NERDatasetInference(Dataset):
    def __init__(self, file_path, tokenizer, block_size = 102, debug = False, debug_num_exmaple = 100, verbose = False):
        self.tokenizer = tokenizer
        self.cls_id = tokenizer.convert_tokens_to_ids('[CLS]')
        self.sep_id = tokenizer.convert_tokens_to_ids('[SEP]')
        
        X = load_data(file_path) # 3-d list
        all_data = []
        
        print('convert tokens to ids ...')        
        for x in X:
            data = {
                'input_ids': self.tokenizer.convert_tokens_to_ids(x),   
                'seq_str': ' '.join(x),
            }
            
            all_data.append(data)
            
            if debug and len(all_data) == debug_num_exmaple:
                break
                
        print('done.')    
            
        self.data = all_data  # list not tensor

        self.process(block_size)
        
        print('num of example : {}'.format(len(self.data)))
        
        if verbose:
            print('one example from {} : '.format(file_path))
            print(self.data[5])
            

    def process(self, block_size):
        text_block_size = block_size - 2
        print('processing padding ...')
        for d in self.data:                
            d['input_ids'] = d['input_ids'][:text_block_size]
            d['input_ids'] = [self.cls_id] + d['input_ids'] + [self.sep_id]
            d['input_ids'] = d['input_ids'] + [0] * (block_size - len(d['input_ids'])) 

        print('done')

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
    

if __name__ == '__main__':
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese", cache_dir = './models/bert/')
    dataset = NERDataset('data/train.txt', tokenizer, block_size=102, debug = False, debug_num_exmaple = 100, verbose = True)
    
    print('-'*30)
    dataset = NERDataset('data/dev.txt', tokenizer, block_size=102, debug = False, debug_num_exmaple = 100, verbose = True)
    print('-'*30)
    dataset = NERDatasetInference('data/test.txt', tokenizer, block_size=102, debug = False, debug_num_exmaple = 100, verbose = True)


