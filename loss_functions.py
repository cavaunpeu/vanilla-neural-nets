from abc import ABCMeta, abstractmethod

import numpy as np


class BaseLossFunction(metaclass=ABCMeta):

    def __init__(self, y_true, y_predicted):        
        self.y_true = y_true
        self.y_predicted = y_predicted

    @property
    def accuracy(self):
        true_labels = np.argmax(self.y_true, axis=1)
        predicted_labels = np.argmax(self.y_predicted, axis=1)        
        return (predicted_labels == true_labels).mean()

    @property
    @abstractmethod
    def cost(self):
        pass

    @property
    @abstractmethod
    def derivative_of_loss_function(self):
        pass


class MeanSquaredError(BaseLossFunction):

    @property
    def cost(self):
        return (.5*(self.y_true - self.y_predicted)**2).sum()
    
    @property
    def derivative_of_loss_function(self):
        return self.y_predicted - self.y_true


class CrossEntropyLoss(BaseLossFunction):
    
    @property
    def cost(self):
        return -(self.y_true*np.log(self.y_predicted) + (1 - self.y_true)*np.log(1 - self.y_predicted)).sum()

    @property
    def derivative_of_loss_function(self):
        return -((self.y_true/self.y_predicted) - ((1 - self.y_true)/(1 - self.y_predicted)))
