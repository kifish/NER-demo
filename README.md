# NER-demo



## Method
1.HMM  
https://github.com/kifish/NER-demo/tree/hmm  
2.CRF  
https://github.com/kifish/NER-demo/tree/crf  
3.BiLSTM-viterbi  
https://github.com/kifish/NER-demo/tree/BiLSTM-viterbi  
4.BiLSTM-CRF  
https://github.com/kifish/NER-demo/tree/BiLSTM-crf  
5.BiLSTM-CNN-CRF   
https://github.com/kifish/NER-demo/tree/BiLSTM-cnn-crf  
Update:        
6.BERT-Softmax               
https://github.com/kifish/NER-demo/tree/bert         
7.BERT-CRF              
https://github.com/kifish/NER-demo/tree/bert


### See more
http://nlpprogress.com/english/named_entity_recognition.html


## Experiment
#### Environment
python3.6+  (according to the branch)    
pip install -r requirements.txt (according to the branch)      
(use pip install git+https://www.github.com/keras-team/keras-contrib.git to install keras-contrib)     

## Result
```
	            precision	   recall        f1-score	   support
BERT                   0.9458      0.9090          0.9263             6195
BERT-CRF               0.9338      0.8901          0.9097             6195
BiLSTM-CRF	       0.8616      0.7138	   0.7806	      6181
BiLSTM-CNN-CRF	       0.8406	   0.7185	   0.7686	      6181
CRF	               0.8420	   0.6279	   0.7170	      6181
BiLSTM-viterbi	       0.8512	   0.5700	   0.6809	      6181
HMM	               0.4911	   0.4341	   0.4479	      6181
```

注: bert做数据预处理的实现和之前的模型不一样, 有一些出入, 导致support有差异, 待对齐。



## Analysis
Todo
