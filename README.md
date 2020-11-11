# NER-demo

文档:     
https://docs.google.com/document/d/e/2PACX-1vTFWjz1j8JQa7lmfl8PNfRc8X_fZ--3yUkeZbYeDqAn2NouBlhO9iKgfzO1uJnaLv8nJcsr61u29ioP/pub


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
6.BERT-Softmax           
https://github.com/kifish/NER-demo/tree/bert              
7.BERT-CRF             
https://github.com/kifish/NER-demo/tree/bert


### See more
http://nlpprogress.com/english/named_entity_recognition.html


## Experiment
#### Environment
sh setup_environment.sh

### Result
bert-softmax

https://github.com/kifish/NER-demo/blob/bert/src/config/config.py

验证集的效果如下: (测试集无label)
```
              precision    recall  f1-score   support

         LOC     0.9085    0.8797    0.8939      2877
         ORG     0.8184    0.7455    0.7803      1336
         PER     0.9234    0.8698    0.8958      1982

   micro avg     0.8945    0.8476    0.8705      6195
   macro avg     0.8834    0.8317    0.8566      6195
weighted avg     0.8938    0.8476    0.8700      6195

              precision    recall  f1-score   support

       B-LOC     0.4620    0.8978    0.6101      2877
       I-LOC     0.5869    0.9342    0.7209      4394
       B-ORG     0.1881    0.7941    0.3042      1331
       I-ORG     0.4646    0.8739    0.6066      5670
       B-PER     0.2041    0.9042    0.3331      1973
       I-PER     0.5518    0.9823    0.7066      3851

   micro avg     0.4108    0.9090    0.5659     20096
   macro avg     0.4096    0.8978    0.5469     20096
weighted avg     0.4638    0.9090    0.6044     20096
```


https://github.com/kifish/NER-demo/blob/bert/src/config/config_v2.py

验证集的效果如下: (测试集无label)
```
              precision    recall  f1-score   support

         LOC     0.9102    0.8669    0.8880      2877
         ORG     0.8149    0.7380    0.7745      1336
         PER     0.8912    0.7356    0.8060      1982

   micro avg     0.8840    0.7971    0.8383      6195
   macro avg     0.8721    0.7802    0.8228      6195
weighted avg     0.8836    0.7971    0.8373      6195

              precision    recall  f1-score   support

       B-LOC     0.9619    0.8863    0.9226      2877
       I-LOC     0.9295    0.9060    0.9176      4394
       B-ORG     0.9276    0.8084    0.8639      1331
       I-ORG     0.9337    0.8670    0.8991      5670
       B-PER     0.9793    0.7922    0.8759      1973
       I-PER     0.8967    0.9873    0.9398      3851

   micro avg     0.9319    0.8901    0.9105     20096
   macro avg     0.9381    0.8745    0.9032     20096
weighted avg     0.9338    0.8901    0.9097     20096
```


### Analysis
Todo