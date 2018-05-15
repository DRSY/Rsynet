from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from layers import Linear,Softmax,Tanh,Dropout,Relu,Sigmod
from nn import NetWork
from loss import Cross_emptropy
from train import train,evaluate
from keras.utils import to_categorical
import numpy as np



iris = load_iris()
X = iris.data
y = iris.target
y = to_categorical(y).astype(np.int)

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.25)

net = NetWork([
    Linear(input_dim=4, output_dim=5),
    Sigmod(),
    Linear(input_dim=5, output_dim=3),
    Dropout(p=0.3),
    Softmax(input_dim=3)
])

train(net, X_train, y_train, epochs=2000, loss=Cross_emptropy(), show_info=True)
acc = evaluate(net, X_test, y_test)
print('acc:{:.2f}%'.format(acc*100))
