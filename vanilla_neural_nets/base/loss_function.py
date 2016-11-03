from abc import ABCMeta, abstractmethod

import numpy as np


class BaseLossFunction(metaclass=ABCMeta):

    @classmethod
    def accuracy(cls, y_true, y_predicted):
        true_labels = np.argmax(y_true, axis=1)
        predicted_labels = np.argmax(y_predicted, axis=1)
        return (predicted_labels == true_labels).mean()

    @classmethod
    @abstractmethod
    def loss(cls):
        pass

    @classmethod
    @abstractmethod
    def total_loss(cls):
        pass

    @classmethod
    @abstractmethod
    def derivative_of_loss_function(cls):
        pass
