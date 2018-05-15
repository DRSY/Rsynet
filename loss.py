import numpy as np

from tensor import Tensor


class Loss:

    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        raise NotImplementedError

    def grad(self, predcited: Tensor, actual: Tensor) -> Tensor:
        raise NotImplementedError

class MSE(Loss):
    '''
    Mean Squared error loss
    '''
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return np.sum((predicted - actual) ** 2)

    def grad(self, predcited: Tensor, actual: Tensor) -> Tensor:
        return 2 * (predcited - actual)


class Cross_emptropy(Loss):
    '''
    Cross emptropy loss
    '''
    def loss(self, predicted: Tensor, actual: Tensor) -> float:
        return -np.sum(actual * np.log(predicted))

    def grad(self, predcited: Tensor, actual: Tensor) -> Tensor:
        return -(actual * np.reciprocal(predcited))
