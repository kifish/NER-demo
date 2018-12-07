from anago.utils import load_data_and_labels
import os
# x_train, y_train = load_data_and_labels(os.path.abspath('../data/train.txt'))
# x_test, y_test = load_data_and_labels(os.path.abspath('../data/dev.txt'))
x_train, y_train = load_data_and_labels('../data/train.txt')
x_test, y_test = load_data_and_labels('../data/dev.txt')

print(x_train[0])
# print('a' + '\t' + 'b')
