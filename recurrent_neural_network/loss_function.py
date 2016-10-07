import itertools

import numpy as np

from neural_network.loss_function import BaseLossFunction


class CrossEntropyLoss(BaseLossFunction):

    @classmethod
    def loss(cls, y_true, y_predicted):
        N = len( list( itertools.chain( *y_true ) ) )
        total_loss = 0
        for prediction_matrix, labels in zip(y_predicted, y_true):
            row_indices = np.arange( len(labels) )
            column_indices = labels
            total_loss += np.sum([ -np.log(prediction_matrix[row_indices, column_indices]) ])            

        return total_loss / N

    @classmethod
    def derivative_of_loss_function(cls, y_true, y_predicted):
        pass
