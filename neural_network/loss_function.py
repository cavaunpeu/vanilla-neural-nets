from abc import ABCMeta, abstractmethod

import numpy as np


class BaseLossFunction(metaclass=ABCMeta):

    EPSILON = 1e-15

    @classmethod
    def accuracy(cls, y_true, y_predicted):
        true_labels = np.argmax(y_true, axis=1)
        predicted_labels = np.argmax(y_predicted, axis=1)
        return (predicted_labels == true_labels).mean()

    @classmethod
    def _bound_predictions(cls, y_predicted):
        y_predicted = np.maximum(y_predicted, cls.EPSILON)
        y_predicted = np.minimum(y_predicted, 1 - cls.EPSILON)
        return y_predicted

    @classmethod
    @abstractmethod
    def cost(cls):
        pass

    @classmethod
    @abstractmethod
    def derivative_of_loss_function(cls):
        pass


class MeanSquaredError(BaseLossFunction):

    @classmethod
    def cost(cls, y_true, y_predicted):
        return (.5*(y_true - y_predicted)**2).mean()

    @classmethod
    def derivative_of_loss_function(cls, y_true, y_predicted):
        return y_predicted - y_true


class CrossEntropyLoss(BaseLossFunction):

    @classmethod
    def cost(cls, y_true, y_predicted):
        y_predicted = cls._bound_predictions(y_predicted)
        return -( y_true*np.log(y_predicted) + (1 - y_true)*np.log(1 - y_predicted) ).mean()

    @classmethod
    def derivative_of_loss_function(cls, y_true, y_predicted):
        y_predicted = cls._bound_predictions(y_predicted)
        return -( (y_true / y_predicted) - ((1 - y_true) / (1 - y_predicted)) )


class LogLikelihoodLoss(BaseLossFunction):

    @classmethod
    def cost(cls, y_true, y_predicted):
        y_predicted = cls._bound_predictions(y_predicted)
        return ( -np.log(y_predicted) * y_true ).mean()

    @classmethod
    def derivative_of_loss_function(cls, y_true, y_predicted):
        y_predicted = cls._bound_predictions(y_predicted)
        return -1/y_predicted * y_true
