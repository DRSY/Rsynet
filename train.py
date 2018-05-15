from tensor import Tensor
from loss import Loss,MSE
from nn import NetWork
from optimizer import Optimizer,SGD
from data import DataIterator,BatchIterator
import numpy as np



def train(net: NetWork,
          inputs: Tensor,
          targets: Tensor,
          epochs: int = 500,
          loss: Loss = MSE(),
          optimizer: Optimizer = SGD(),
          iterator: DataIterator = BatchIterator(),
          show_info: bool = False):
    for epoch in range(epochs):
        epoch_loss = .0
        for batch_inputs, batch_targets in iterator(inputs, targets):
            predictions = net.forward(batch_inputs)
            epoch_loss += loss.loss(predictions, batch_targets)
            grad = loss.grad(predictions, batch_targets)
            net.backward(grad)
            optimizer.step(net)
        if show_info:
            print('epoch:{},  loss:{}'.format(epoch, epoch_loss))


def evaluate(net: NetWork, inputs: Tensor, targets: Tensor) -> float:
    predictions = net.evaluate(inputs)
    predictions = np.argmax(predictions, axis=1)
    targets = np.argmax(targets, axis=1)
    acc = np.mean(predictions == targets)
    return acc

