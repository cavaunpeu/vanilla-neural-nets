import itertools

import numpy as np

from base.loss_function import BaseLossFunction


class CrossEntropyLoss(BaseLossFunction):

    @classmethod
    def loss(cls, y_true, y_predicted):
        return cls.total_loss(y_true=y_true, y_predicted=y_predicted) / len(y_true)

    @classmethod
    def total_loss(cls, y_true, y_predicted):
        row_indices = np.arange( len(y_true) )
        column_indices = y_true
        return np.sum([ -np.log(y_predicted[row_indices, column_indices]) ])

    @classmethod
    def derivative_of_loss_function(cls, y_true, y_predicted):
        pass
