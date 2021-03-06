import numpy as np


from vanilla_neural_nets.base.loss_function import BaseLossFunction


class MeanSquaredError(BaseLossFunction):

    @classmethod
    def loss(cls, y_true, y_predicted):
        return cls.total_loss(y_true, y_predicted) / len(y_true)

    @classmethod
    def total_loss(cls, y_true, y_predicted):
        return np.sum( (.5*(y_true - y_predicted)**2) )

    @classmethod
    def derivative_of_loss_function(cls, y_true, y_predicted):
        return y_predicted - y_true


class BinaryCrossEntropyLoss(BaseLossFunction):

    @classmethod
    def loss(cls, y_true, y_predicted):
        return cls.total_loss(y_true, y_predicted) / len(y_true)

    @classmethod
    def total_loss(cls, y_true, y_predicted):
        return np.sum( -( y_true*np.log(y_predicted) + (1 - y_true)*np.log(1 - y_predicted) ) )

    @classmethod
    def derivative_of_loss_function(cls, y_true, y_predicted):
        return -( (y_true / y_predicted) - ((1 - y_true) / (1 - y_predicted)) )
