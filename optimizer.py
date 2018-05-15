from nn import NetWork

class Optimizer(object):

    def step(self, net: NetWork) ->None:
        raise  NotImplementedError


class SGD(Optimizer):
    '''
    Stocastic gradient descent
    '''
    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, net: NetWork) -> None:
        for params, grads in net.params_grads():
            params -= self.lr*grads
