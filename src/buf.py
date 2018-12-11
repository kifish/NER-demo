# X = [['a','1'],['b','2']]
# print([word for sent in X for word in sent])
#
#
# x = [1.0,2.0]
# print(x.index(2))
#


#
# dict = {'Name': 'Zara', 'Age': 27}
#
# print("Value : %s" %  dict.get('Age'))
# print("Value : %s" %  dict.get('Sex', 0))
# print(dict.get(0,default=0))
#
from src.utils import load_data_and_labels,save_pred,transformer_x,transformer_y
transformer_y = transformer_y(2)
X = [['我','家'],['家','我']]
Y = [['O','B-ORG'],['B-ORG','O']]
Y = transformer_y.to_onehot(Y)
print(Y)


