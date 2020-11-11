import os, sys
import logging
from trainer import *
from model import *
from dataset import *

from shutil import copytree
from time import strftime, localtime


class Config():
    def __init__(self) -> None:
        self.cache_dir = './models/bert/'
        self.base_model_name = 'bert-base-chinese'
        
        self.trainer = Trainer_v2
        self.model = BertCRFForNER_v3
        self.mode = 'run_all'
        
        self.infer_times = 1 
        self.save_params = True
        self.preprocess_only = False
        self.save_params = self.save_params and (not self.preprocess_only) and self.mode != 'run_test' and self.mode != 'run_val'


        self.data_config = {
            'debug': False,
            'debug_num_exmaple': None,
            'block_size': 30,    
            'verbose': True,
        }
                
        # self.data_config = {
        #     'debug': True,
        #     'debug_num_exmaple': 1000,
        #     'block_size': 30,    
        #     'verbose': True,
        # }
        
        self.dataset = NERDataset 
        self.predict_dataset = NERDatasetInference

        self.train_data_config = self.data_config.copy()
        self.train_data_config['file_path'] = 'data/train.txt'
        
        self.val_data_config = self.data_config.copy()
        self.val_data_config['file_path'] = 'data/dev.txt'
        self.val_data_config['block_size'] = 102


        self.test_data_config = self.data_config.copy()
        self.test_data_config['file_path'] = 'data/test.txt'
        self.test_data_config['block_size'] = 102

        self.use_crf_lib = True
        # train config
        self.sample_train_data = False
        self.shuffle_on_the_fly = True

        self.use_cuda = True # GPU
        self.use_multi_gpus = False # 单机多卡
        
        self.model_config = {
            'base_model_name': self.base_model_name,
            'cache_dir': self.cache_dir,
            'use_cuda': self.use_cuda,
            'hidden_size': 768,
            'hidden_dropout_prob': 0.1,
        }
        
        
        # self.l2_reg = None 
        # self.weight_decay = 1e-4
        self.lr = 2e-5
        self.init_clip_max_norm = 1.0
        
        self.num_epoch = 20
        self.batch_size = 256 
        # batch_size is the total batch_size when use_multi_gpus is set as True
        # 多卡情况下须用偶数, 否则会报错。且需要drop掉最后一个batch(如果最后一个batch的样本数为奇数)

        # print and save
        self.print_every = 5
        self.val_every = 100  # 根据batch size变化
        self.force_save_every = None
        self.val_num = None

        # save and log
        run_name = None
        run_index = 4
        if run_name is None:
            run_name = 'run{}'.format(run_index)
        self.save_dir = 'records/simple/{}'.format(run_name)
        self.save_dir = os.path.abspath(self.save_dir)

        self.check_save_dirs()
        
        self.save_src_and_dst = ('src', os.path.join(self.save_dir, 'src'))
             
        self.log_dir = os.path.join(self.save_dir,'log') 
        self.ckpt_dir = os.path.join(self.save_dir,'checkpoint')
        self.model_save_name = 'model_param.pt'
        self.ckpt_file = os.path.join(self.ckpt_dir, self.model_save_name)
        self.save_info_log_name = 'model_save_info.log'
        self.ckpt_info_log = os.path.join(self.ckpt_dir, self.save_info_log_name)
        current_time = strftime("%y%m%d-%H%M", localtime())
        self.log_file = '{}.log'.format(current_time)
        self.log_file = os.path.join(self.log_dir, self.log_file)   
        self.generated_results_save_path = os.path.join(self.log_dir, current_time + '_generated_results.json')  
        self.tensorboard_dir = os.path.join(self.log_dir,'tensorboard')      
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s",
                              "%Y-%m-%d %H:%M:%S")

        # check dirs
        self.check_dirs()

        if self.save_params:
            self.save_all_program_files()
        
        if not self.preprocess_only:
            std_h = logging.StreamHandler(sys.stdout)
            std_h.setFormatter(formatter)
            self.logger.addHandler(std_h)
            
            file_h = logging.FileHandler(self.log_file)  # 需要目录已经存在            
            file_h.setFormatter(formatter)
            self.logger.addHandler(file_h)

    def check_save_dirs(self):
        if self.mode == 'run_all' or self.mode == 'run_train':
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            else:
                print('{} was used'.format(self.save_dir))
                save_parent_dir = os.path.split(self.save_dir[:-1])[0]
                paths = [p for p in os.listdir(save_parent_dir) if p[:3] == 'run']
                max_num = 1
                for p in paths:
                    num_suffix = p[3:]
                    if num_suffix:
                        num = int(num_suffix)
                    else:
                        num = 1
                    max_num = max(max_num, num)
                new_num_suffix = str(max_num + 1)
                self.save_dir = os.path.join(save_parent_dir, 'run' + new_num_suffix)
                print('now use {}'.format(self.save_dir))
                os.makedirs(self.save_dir)
        else:
            if not os.path.exists(self.save_dir):
                raise Exception('{} does not exist'.format(self.save_dir))

    def check_dirs(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        if self.preprocess_only or self.mode == 'run_test':
            return

        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

    def check_params(self):
        assert type(self.infer_times) == int
        assert self.infer_times >= 1 and self.infer_times <= 10

    def save_all_program_files(self):
        src, dst = self.save_src_and_dst
        copytree(src, dst)


        



        