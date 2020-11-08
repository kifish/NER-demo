import json
import torch, time, os
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import optim
from tqdm import tqdm
from transformers import AutoTokenizer
from utils import *
    

batchfy_fn = batchfy_wrapper(torch.device('cuda'), with_tag = True)
test_batch_fn = batchfy_wrapper(torch.device('cuda'), with_tag = False)

class Trainer:
    def __init__(self, config):
        # tokenizer
        self.tokenizer =  AutoTokenizer.from_pretrained(config.base_model_name, cache_dir = config.cache_dir)
        config.logger.info('vocab size : {}'.format(len(self.tokenizer)))
        
        self.tag_encoder = TagEncoder()
        
        d = {'tokenizer': self.tokenizer}
        # data
        if config.mode == 'run_all' or config.mode == 'run_train':
            
            config.train_data_config.update(d)
            config.val_data_config.update(d)
            config.test_data_config.update(d)

            
            self.trainset = config.dataset(**config.train_data_config)
            self.valset = config.dataset(**config.val_data_config)
            
            self.testset = config.predict_dataset(**config.test_data_config)
            
        elif config.mode == 'run_val':
            config.val_data_config.update(d)
            self.valset = config.dataset(**config.val_data_config)
        
        elif config.mode == 'run_test' or config.mode == 'run_predict':
            config.test_data_config.update(d)
            self.testset = config.predict_dataset(**config.test_data_config)


        self.model = config.model(config.model_config)
        
        self.device = torch.device('cuda' if config.use_cuda else 'cpu')
        if config.use_cuda:
            config.logger.info('num of gpus : {}'.format(torch.cuda.device_count()))
            if config.use_multi_gpus:
                self.model = nn.DataParallel(self.model).to(self.device)
                config.logger.info('names of gpus : {}'.format(torch.cuda.get_device_name()))

            else:                
                self.model = self.model.cuda()
                config.logger.info('name of gpus : {}'.format(torch.cuda.get_device_name()))
            
        self.config = config
        
        if config.mode == 'run_all' or config.mode == 'run_train':
            self.writer = SummaryWriter(self.config.tensorboard_dir)

        self.criterion = nn.CrossEntropyLoss(ignore_index= 0, reduction = 'mean') 
        self.seq_criterion = nn.CrossEntropyLoss(ignore_index= 0, reduction = 'sum') 

        self.optimizer = optim.Adam(self.model.parameters(), lr = self.config.lr)

        self.config.logger.info(self.model)
        

    def save_checkpoint(self, save_path, ckpt_info_log_path, save_info, save_optim = False):
        # DataParallel wrappers keep raw model object in .module attribute
        raw_model = self.model.module if hasattr(self.model, "module") else self.model
        
        if save_optim:
            torch.save({
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                }, save_path) # 会占用大量空间
        
        else:
            # torch.save(raw_model.state_dict(), save_path)
            
            torch.save({
                'model_state_dict': raw_model.state_dict(),
                }, save_path) 

        self.config.logger.info('saved model into {} \n'.format(save_path))
        
        # log
        with open(ckpt_info_log_path,'w') as f:
            f.write(save_info + '\n')
    
    
    # https://stackoverflow.com/questions/735975/static-methods-in-python
    # https://www.geeksforgeeks.org/class-method-vs-static-method-python/
    @staticmethod
    def __acc(pred, label, skipped_class = [0, 7]):
        # pred: 2-d tensor; label: 1-d tensor
        # 0是padding; 7是O
        def argmax(l):
            return max(enumerate(l), key=lambda x: x[1])[0]   
        
        pred = pred.cpu().detach().numpy().tolist() # list; pytorch 1.7
        label = label.cpu().detach().numpy().tolist() # list
        
        pred = list(map(argmax, pred)) # -> 1-d list
        
        real_padding_ratio = sum(map(lambda x: x == 0, label)) / len(label)
        pred_padding_ratio = sum(map(lambda x: x == 0, pred)) / len(pred)
        
        correct = [ p == t for p,t in zip(pred, label)]
        if len(correct):
            raw_acc = sum(correct) / len(correct) # 没做mask;里面有很多padding tag; 而模型一开始会全部预测为O, 之后会预测为实体的tag,基本不会预测为padding tag; 因此raw_acc会很低
        else:
            raw_acc = 0
        
        target_correct = []
        pure_target_correct = []

        for p, t in zip(pred, label):
            if t == 0: # padding
                continue
            hit = p == t
            target_correct.append(hit)

            if t in skipped_class: # padding or O
                continue
            pure_target_correct.append(hit)
            

        if len(target_correct):
            target_acc = sum(target_correct) / len(target_correct) 
        else:
            target_acc = 0
            
        if len(pure_target_correct):
            pure_target_acc = sum(pure_target_correct) / len(pure_target_correct)
        else:
            pure_target_acc = 0
        
        return raw_acc, target_acc, pure_target_acc,real_padding_ratio,pred_padding_ratio
    

    def train(self, train_data_loader, val_data_loader):
        best_result = {'loss':None}
        global_step = 0
        
        train_loss = 0 # init
        raw_acc, target_acc, pure_target_acc = 0, 0, 0 # init
        real_padding_ratio, pred_padding_ratio = 0, 0 # init
        
        for epoch in tqdm(range(self.config.num_epoch)):
            self.config.logger.info('>' * 100)
            self.config.logger.info('epoch: {}'.format(epoch + 1))
            n_step, n_sample_total, loss_total = 0, 0, 0 # in the epoch
            raw_acc_total, target_acc_total, pure_target_acc_total = 0,0,0 # init
            real_padding_ratio_total = 0
            pred_padding_ratio_total = 0
            
            print_cnt = 0
            seq_loss_total = 0
            start_time = time.time()

            # switch model to training mode
            self.model.train()

            for batch_idx, batch_samples in enumerate(tqdm(train_data_loader)):
                global_step += 1
                n_step += 1
                
                # batch
                input_ids, attention_mask, target_ids = batch_samples
                n_sample_total += input_ids.size(0) # 实际的样本个数
                
                ############# train  model #####################
                self.optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask = attention_mask) # type_ids 会自动补全为0 
                pred_scores = outputs[0] # logits
                # pred_scores: (b,seq_len, class)
                # target_ids: (b,seq_len)
    
                pred_scores = pred_scores.view(-1, pred_scores.size(-1))
                target_ids = target_ids.view(pred_scores.size(0))
                
                loss = self.criterion(pred_scores, target_ids) # 做了mask; mean token loss; backward 不要用 mean seq loss
                loss.backward()
                
                if self.config.init_clip_max_norm is not None:                    
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], \
                            max_norm = self.config.init_clip_max_norm)
                    if grad_norm >= 1e2:
                        self.config.logger.info('WARNING: Exploding Gradients {:.2f}'.format(grad_norm))
            
                self.optimizer.step()
                ####################################################
                
                loss_total += loss.item() 
                train_loss = loss_total / n_step # train_mean_token_loss in the epoch
                
                self.writer.add_scalar('Train/Loss', train_loss , global_step)

                if global_step % self.config.print_every == 0:
                    print_cnt += 1 # 这里为了代码的简洁重复计算了loss, 追求高效的话可以手动计算每个batch实际的需要预测的token数量和实际的句子个数
                    pred_scores = pred_scores.view(-1, pred_scores.size(-1))
                    target_ids = target_ids.view(pred_scores.size(0))
                    
                    seq_loss_total += self.seq_criterion(pred_scores, target_ids).item() / input_ids.size(0)
                    train_seq_loss = seq_loss_total / print_cnt

                    b_raw_acc, b_target_acc, b_pure_target_acc,b_real_padding_ratio,b_pred_padding_ratio = self.__acc(pred_scores, target_ids)
                    # 如果最后一个batch的bsz变小, 结果会有出入
                    raw_acc_total += b_raw_acc
                    target_acc_total += b_target_acc
                    pure_target_acc_total += b_pure_target_acc
                    
                    raw_acc = raw_acc_total / print_cnt
                    target_acc = target_acc_total / print_cnt
                    pure_target_acc = pure_target_acc_total / print_cnt
                    
                    real_padding_ratio_total += b_real_padding_ratio
                    pred_padding_ratio_total += b_pred_padding_ratio
                    
                    real_padding_ratio = real_padding_ratio_total / print_cnt
                    pred_padding_ratio = pred_padding_ratio_total / print_cnt

                    self.config.logger.info(
                        'epoch {}, iteration {}, '
                        'train_mean_token_loss: {:.4f}, '
                        'train_mean_seq_loss: {:.4f}, '
                        'train_mean_raw_acc: {:.4f}, '
                        'train_mean_target_acc: {:.4f}, '
                        'train_mean_pure_target_acc: {:.4f}, '
                        'real_padding_ratio: {:.4f}, '
                        'pred_padding_ratio: {:.4f}'
                        .format(epoch + 1, batch_idx + 1, train_loss, train_seq_loss,
                               raw_acc, target_acc, pure_target_acc,real_padding_ratio,pred_padding_ratio))
                
                
                    self.writer.add_scalar('Train/real_padding_ratio', real_padding_ratio , global_step)
                    self.writer.add_scalar('Train/pred_padding_ratio', pred_padding_ratio , global_step)
                    self.writer.add_scalar('Train/raw_acc', raw_acc , global_step)
                    self.writer.add_scalar('Train/target_acc', target_acc , global_step)
                    self.writer.add_scalar('Train/pure_target_acc', pure_target_acc , global_step)


                # val
                if global_step % self.config.val_every == 0 or \
                        (self.config.force_save_every is not None and
                            global_step % self.config.force_save_every == 0):
                    
                    self.config.logger.info('epoch {}, iteration {}, validating...'.format(epoch + 1, batch_idx + 1))
                    val_result = self.inference(val_data_loader, mode = 'val')
                    flatten_result = self.flatten_dict(val_result)
                    for k,v in flatten_result.items():
                        self.writer.add_scalar('Val/{}'.format(k), v, global_step) 

                    # remember
                    self.model.train()
                    
                    # save_best_only
                    if best_result['loss'] is None or val_result['loss'] < best_result['loss']:
                        best_result = val_result
                        
                        save_info = 'save info : epoch_{}_iteration_{}_val_info:\n'. \
                            format(epoch + 1, batch_idx + 1)
                            
                        # https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable/50916741
                        save_info += json.dumps(val_result, indent=4, sort_keys= True) # pretty print

                        self.save_checkpoint(self.config.ckpt_file, self.config.ckpt_info_log, save_info)
                        
                    elif (self.config.force_save_every is not None and
                            global_step % self.config.force_save_every == 0):
                        
                        save_info = 'save info : epoch_{}_iteration_{}_val_info:\n'. \
                            format(epoch + 1, batch_idx + 1)
                        
                        save_info += json.dumps(val_result, indent=4, sort_keys= True) # pretty print
                        
                        num = global_step // self.config.force_save_every
                        
                        ckpt_info_log_path = os.path.join(self.config.ckpt_dir,'force_{}_{}'.format(
                            num, self.config.save_info_log_name
                        ))
                            
                        ckpt_force_file = os.path.join(self.config.ckpt_dir,'force_{}_{}'.format(
                            num, self.config.model_save_name
                        ))
                        
                        self.config.logger.info('force save ...')
                        self.save_checkpoint(ckpt_force_file, ckpt_info_log_path, save_info)

            train_seq_loss = 0 # 0 表示未计算
            if print_cnt > 0:
                train_seq_loss = seq_loss_total / print_cnt
            end_time = time.time()
            epoch_mins, epoch_secs = cal_elapsed_time(start_time, end_time)
            self.config.logger.info(
                        'epoch {}, '
                        'train_mean_token_loss: {:.4f}, '
                        'train_mean_seq_loss: {:.4f}, '
                        'train_mean_raw_acc: {:.4f}, '
                        'train_mean_target_acc: {:.4f}, '
                        'train_mean_pure_target_acc: {:.4f},'
                        'train_mean_real_padding_ratio: {:.4f},'
                        'train_mean_pred_padding_ratio: {:.4f},'
                        'time: {}m {}s'.
                        format(epoch + 1, train_loss, train_seq_loss,
                               raw_acc, target_acc, pure_target_acc, 
                               real_padding_ratio,
                               pred_padding_ratio,
                               epoch_mins, epoch_secs))
            

    # https://stackoverflow.com/questions/13183501/staticmethod-and-recursion
    @staticmethod
    def flatten_dict(d, lkey= '', sep = '.'):
        ret = {}
        for rkey,val in d.items():
            key = lkey + rkey
            if isinstance(val, dict):
                ret.update(Trainer.flatten_dict(val, key + sep, sep))
            else:
                ret[key] = val
        return ret


    def inference(self, data_loader, mode = 'val'):
        n_samples, loss_total = 0, 0
        seq_loss_sum = 0
        n_batch = 0
        self.model.eval()
        y_test = [] # 2-d list;[[int]]
        pred = [] # 2-d list;[[int]]
        
        raw_acc_total, target_acc_total, pure_target_acc_total = 0, 0, 0 # init
        real_padding_ratio_total = 0
        pred_padding_ratio_total = 0
        with torch.no_grad():
            for batch_idx, batch_samples in enumerate(data_loader):
                n_batch += 1
                input_ids, attention_mask, target_ids = batch_samples
                
                the_real_batch_size = input_ids.size(0)
                n_samples += the_real_batch_size
                
                mask = target_ids != 0
                the_real_num_target_id = mask.sum().item()
                
                outputs = self.model(input_ids, attention_mask = attention_mask) 
                pred_scores = outputs[0] # 3-d
                
                y_test += target_ids.cpu().numpy().tolist()
                pred += torch.argmax(pred_scores, dim=-1).cpu().numpy().tolist()
                
                # reshape
                pred_scores = pred_scores.view(-1, pred_scores.size(-1))
                target_ids = target_ids.view(pred_scores.size(0))
                
                loss = self.criterion(pred_scores, target_ids)
                loss_total += loss.item()
                
                seq_loss_sum += loss.item() * the_real_num_target_id
                
                b_raw_acc, b_target_acc, b_pure_target_acc, b_real_padding_ratio, b_pred_padding_ratio = self.__acc(pred_scores, target_ids)
                raw_acc_total += b_raw_acc
                target_acc_total += b_target_acc
                pure_target_acc_total += b_pure_target_acc
                
                real_padding_ratio_total += b_real_padding_ratio
                pred_padding_ratio_total += b_pred_padding_ratio
            
                if self.config.val_num is not None and batch_idx + 1 == self.config.val_num:
                    break
        
        mean_token_loss = loss_total / n_batch
        mean_seq_loss = seq_loss_sum / n_samples
        
        raw_acc = raw_acc_total / n_batch
        target_acc = target_acc_total / n_batch
        pure_target_acc = pure_target_acc_total / n_batch
        
        real_padding_ratio = real_padding_ratio_total / n_batch
        pred_padding_ratio = pred_padding_ratio_total / n_batch
       
        result = {
            'loss' : mean_token_loss,
            'seq_loss' : mean_seq_loss,
            'raw_acc': raw_acc,
            'target_acc': target_acc,
            'pure_target_acc': pure_target_acc,
            'real_padding_ratio': real_padding_ratio,
            'pred_padding_ratio': pred_padding_ratio,
        }
        
        
        eval_result, result1_str, result2_str = self.eval(y_test, pred)
        
        part_result = result.copy()
        
        result.update(eval_result)
        
        # time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # self.config.logger.info('time : {}\n'.format(time_str))
        self.config.logger.info("the whole/part of {} dataset:".format(mode))
        self.config.logger.info("n_samples : {}".format(n_samples))
        self.config.logger.info('{} info:\n{}'.format(mode, json.dumps(part_result, indent = 4, sort_keys = True)))
        self.config.logger.info('{} info:\n{}'.format(mode, result1_str))
        self.config.logger.info('{} info:\n{}'.format(mode, result2_str))
        
        return result



    def eval(self, y_test_ids, pred_ids):
        y_test = self.tag_encoder.to_tag(y_test_ids) # 模型一开始会全部预测为O
        pred = self.tag_encoder.to_tag(pred_ids)
        return eval(y_test, pred, y_test_ids, pred_ids)
    
    # todo
    def predict(self, data_loader):
        pass 
    
    
    def batch_predict(self, batch_input):
        
        pass 

    def single_predict(self, text):
        
        pass 
    

    def run(self,mode = 'run_all'):
        self.config.logger.info("mode : {}".format(mode))
        
        if mode == 'run_all': 
            need_drop = False
            if self.config.use_cuda and self.config.use_multi_gpus:
                need_drop = True

            if self.config.sample_train_data:
                raise NotImplementedError
            else:
                shuffle = False
                if self.config.shuffle_on_the_fly:
                    shuffle = True
                    self.config.logger.info("shuffle_on_the_fly")
                
                train_data_loader = DataLoader(dataset = self.trainset, batch_size = self.config.batch_size, \
                    shuffle = shuffle, collate_fn=batchfy_fn, drop_last = need_drop) 

            val_data_loader = DataLoader(dataset = self.valset, \
                batch_size = self.config.batch_size * self.config.infer_times, \
                shuffle = False, collate_fn= batchfy_fn, drop_last = need_drop)
            test_data_loader = DataLoader(dataset = self.testset, \
                batch_size = self.config.batch_size * self.config.infer_times, \
                    shuffle = False, collate_fn= batchfy_fn, drop_last = need_drop)

            self.train(train_data_loader, val_data_loader)
            self.model.load_state_dict(torch.load(self.config.ckpt_file)['model_state_dict'])
            self.inference(val_data_loader, mode = 'val')
            # self.inference(test_data_loader, mode = 'test')
            
        elif mode == 'run_val':
            val_data_loader = DataLoader(dataset = self.valset, \
                batch_size = self.config.batch_size * self.config.infer_times, \
                 shuffle=False, collate_fn= batchfy_fn)
            self.model.load_state_dict(torch.load(self.config.ckpt_file)['model_state_dict'])
            self.inference(val_data_loader, mode = 'val')
            
            
        elif mode == 'run_test':
            
            test_data_loader = DataLoader(dataset = self.testset, \
                batch_size = self.config.batch_size * self.config.infer_times, \
                 shuffle=False, collate_fn = batchfy_fn)
            
            self.model.load_state_dict(torch.load(self.config.ckpt_file)['model_state_dict'])
            self.config.logger.info("loaded the trained model.")
            self.inference(test_data_loader, mode = 'test')
            
            
        elif mode == 'run_predict':
            test_data_loader = DataLoader(dataset = self.testset, \
                batch_size = self.config.batch_size * self.config.infer_times, \
                 shuffle=False, collate_fn = test_batch_fn)
            
            self.model.load_state_dict(torch.load(self.config.ckpt_file)['model_state_dict'])
            self.config.logger.info("loaded the trained model.")
            self.predict(test_data_loader, mode = 'test')    
                        
            
            
            
            
            

