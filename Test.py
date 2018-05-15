from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from nn import NetWork
from layers import Layer,Linear,Dropout,Softmax,Tanh
from loss import Cross_emptropy
from train import train,evaluate
from keras.utils import to_categorical
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3)

y_train_one_hot = to_categorical(y_train).astype(np.int)
y_test_one_hot = to_categorical(y_test).astype(np.int)

net = NetWork([
    Linear(input_dim=4, output_dim=5),
    Tanh(),
    Linear(input_dim=5, output_dim=3),
    Dropout(p=0.3),
    Softmax(input_dim=3)
])

svc = SVC()
lr = LogisticRegression()

# 训练SVM分类器
svc.fit(X_train, y_train)

# 训练逻辑斯蒂回归分类器
lr.fit(X_train, y_train)

# 训练神经网络
train(net, X_train, y_train_one_hot, epochs=2000, loss=Cross_emptropy(), show_info=False)

pred_svc = svc.predict(X_test)
acc_svc = accuracy_score(y_test, pred_svc)

pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, pred_lr)

acc_nn = evaluate(net, X_test, y_test_one_hot)

print('accuracy of SVM:', acc_svc)
print('accuracy of Logistic Regression:', acc_lr)
print('accuracy of Nurual Network:', acc_nn)







