# from anago.utils import load_data_and_labels  #这里是靠空行来切割句子的，这会导致句子过于长。
import re
def load_data_and_labels(path):
    split_pattern = re.compile(r'[；;。，、？！\.\?,! ]')
    x = []
    y = []
    sentence = []
    labels = []
    with open(path,'r',encoding = 'utf8') as f:
        for line in f.readlines():#每行为一个字符和其tag，中间用tab隔开
            line = line.strip().split('\t')
            if(not line or len(line) < 2): continue
            word, tag= line[0], line[1]
            if split_pattern.match(word):
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
    return x,y

x_train, y_train = load_data_and_labels('../data/train.txt')
x_test, y_test = load_data_and_labels('../data/dev.txt')

print(x_train[0])
print(y_train[0])

import anago
model = anago.Sequence()
model.fit(x_train, y_train, epochs=1)

print(model.score(x_test, y_test))

