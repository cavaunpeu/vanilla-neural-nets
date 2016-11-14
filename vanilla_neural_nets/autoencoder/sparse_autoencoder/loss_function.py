import numpy as np

from vanilla_neural_nets.base.loss_function import BaseLossFunction


class SparseMeanSquaredError(BaseLossFunction):

    @classmethod
    def loss(cls, y_true, y_predicted):
        return y_predicted / len(y_true)

    @classmethod
    def total_loss(cls, y_true, y_predicted):
        return np.sum(y_predicted)

    @classmethod
    def derivative_of_loss_function(cls, y_true, y_predicted):
        return y_predicted
