import numpy as np

from tensor import Tensor
from typing import Callable

np.random.seed(42)

class Layer:

    def __init__(self) -> None:
        self.params = {}
        self.grads = {}


    def forward(self, inputs: Tensor, train: bool = True) -> Tensor:
        """
        produce output
        :param input:
        :return:
        """
        raise  NotImplementedError

    def backward(self, grad: Tensor) -> Tensor:
        raise NotImplementedError


class Linear(Layer):
    '''
    线性层
    output = input * W + bais
    '''

    def __init__(self, input_dim: int, output_dim: int) -> None:
        '''
        inputs -> (batch_size, input_dim)
        outputs -> (batch_size, output_dim)
        '''
        super().__init__()
        self.params["w"] = np.random.uniform(-1/np.sqrt(input_dim), 1/np.sqrt(input_dim), size=(input_dim, output_dim))
        self.params["b"] = np.random.randn(output_dim)


    def forward(self, inputs: Tensor, train: bool = True) -> Tensor:
        self.inputs = inputs
        return self.inputs @ self.params["w"] + self.params["b"]

    def backward(self, grad: Tensor) -> Tensor:

        '''
        y = f(x), x = a @ b + c
        dy/da = f'(x) @ b.T
        dy/db = a.T @ f'(x)
        '''
        self.grads["b"] = np.sum(grad, axis=0)
        self.grads["w"] = self.inputs.T @ grad
        return grad @ self.params["w"].T


F = Callable[[Tensor], Tensor]
class Activation(Layer):

    def __init__(self, f: F, f_prime: F) -> None:
        super().__init__()
        self.f = f
        self.f_prime = f_prime

    def forward(self, inputs: Tensor, train: bool = True) -> Tensor:
        self.inputs = inputs
        return self.f(self.inputs)

    def backward(self, grad: Tensor) -> Tensor:
        return self.f_prime(self.inputs) * grad


def tanh(x: Tensor) -> Tensor:
    return np.tanh(x)

def tanh_prime(x: Tensor) -> Tensor:
    y = tanh(x)
    return 1 - y ** 2

def sigmod(x: Tensor) -> Tensor:
    return 1/(1 + np.exp(-x))

def sigmod_prime(x: Tensor) -> Tensor:
    return sigmod(x) * (1 - sigmod(x))


def relu(x: Tensor) -> Tensor:
    return x * (x>0)

def relu_prime(x: Tensor) -> Tensor:
    return (x>0).astype(np.int)



class Tanh(Activation):

    def __init__(self):
        super().__init__(tanh, tanh_prime)


class Sigmod(Activation):

    def __init__(self):
        super().__init__(sigmod, sigmod_prime)


class Relu(Activation):

    def __init__(self):
        super().__init__(relu, relu_prime)


def softmax(x):
    x = x.T
    ret = np.exp(x) / np.sum(np.exp(x), axis=0)
    return ret.T


class Softmax(Layer):
    '''
    Softmax layer
    '''
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim

    def forward(self, inputs: Tensor, train: bool = True) -> Tensor:
        self.inputs = inputs
        self.grads = np.zeros((self.inputs.shape[0], self.input_dim, self.input_dim), dtype=np.float)
        self.outputs = softmax(self.inputs)
        return self.outputs

    def backward(self, grad: Tensor) -> Tensor:
        for i in range(self.grads.shape[0]):
            for j in range(self.input_dim):
                for k in range(self.input_dim):
                    if k == j:
                        self.grads[i][j][k] = self.outputs[i][k]*(1-self.outputs[i][k])
                    else:
                        self.grads[i][j][k] = - self.outputs[i][k]*self.outputs[i][j]
        ret = np.zeros((self.inputs.shape[0], self.input_dim), dtype=np.float)
        for i in range(self.inputs.shape[0]):
            ret[i] = grad[i] @ self.grads[i]
        return ret

class Dropout(Layer):
    '''
    Dropout layer
    不含实际参数W和b，只在前向传播和反向传播时
    '''
    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        self.p = p
        self.prob = []

    def forward(self, inputs: Tensor, train: bool = True) -> Tensor:
        self.inputs = inputs
        self.train = train
        if self.train:
            # 训练时
            self.prob = np.array([1 if (np.random.randint(0,10)/10)>self.p else 0 for _ in range(self.inputs.shape[1])], dtype=np.float)
            self.inputs *= self.prob
        else:
            # 测试时，权重预乘以p
            self.inputs *= self.p
        return self.inputs

    def backward(self, grad: Tensor) -> Tensor:
        '''
        grad -> (batch_size, output_dim)
        return -> (batch_size, output_dim)
        '''
        grad *= self.prob
        return grad




