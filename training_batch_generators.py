from abc import ABCMeta, abstractmethod

import numpy as np

from data_objects import TrainingBatch


class BaseTrainingBatchGenerator(metaclass=ABCMeta):
    
    def __init__(self, X, y, n_batches):
        self.X = X
        self.y = y
        self.n_batches = n_batches
        
    @abstractmethod
    def __iter__(self):
        pass


class MiniBatchGenerator(BaseTrainingBatchGenerator):
        
    def __iter__(self):
        for batch_index in np.array_split(range(len(self.X)), self.n_batches):
            yield TrainingBatch(X=self.X[batch_index], y=self.y[batch_index])
