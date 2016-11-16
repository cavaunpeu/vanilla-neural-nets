import numpy as np

from vanilla_neural_nets.base.loss_function import BaseLossFunction
from vanilla_neural_nets.neural_network.loss_function import MeanSquaredError


class SparseMeanSquaredError(BaseLossFunction):

    @classmethod
    def loss(cls, y_true, y_predicted, rho, vector_of_rho_hats, beta):
        return MeanSquaredError.loss(y_true, y_predicted) + \
            beta * KLDivergenceSparsityLoss.loss(rho, vector_of_rho_hats)

    @classmethod
    def total_loss(cls, y_true, y_predicted, rho, vector_of_rho_hats, beta):
        return MeanSquaredError.total_loss(y_true, y_predicted) + \
            beta * KLDivergenceSparsityLoss.total_loss(rho, vector_of_rho_hats, y_true)

    @classmethod
    def derivative_of_loss_function(cls, y_true, y_predicted):
        """We only ever compute the derivative of the loss function with respect
        to the inputs to our output layer. The sparsity term is not part of this
        computation."""
        return MeanSquaredError.derivative_of_loss_function(y_true, y_predicted)


class KLDivergenceSparsityLoss(BaseLossFunction):

    @classmethod
    def loss(cls, rho, vector_of_rho_hats):
        first_term = rho * np.log( rho / vector_of_rho_hats )
        second_term = (1 - rho) * np.log( (1 - rho) / (1 - vector_of_rho_hats) )
        return (first_term + second_term).sum()

    @classmethod
    def total_loss(cls, rho, vector_of_rho_hats, y_true):
        return y_true.shape[0] * cls.loss(rho, vector_of_rho_hats)

    @classmethod
    def derivative_of_loss_function(cls, rho, vector_of_rho_hats):
        return - (rho / vector_of_rho_hats) + (1 - rho) / (1 - vector_of_rho_hats)
