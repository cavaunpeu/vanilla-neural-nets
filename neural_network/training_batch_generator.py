from abc import ABCMeta, abstractmethod

import numpy as np

from neural_network.data_object import TrainingBatch


class BaseTrainingBatchGenerator(metaclass=ABCMeta):

    def __init__(self, X, y, batch_size, random_number_generator):
        shuffled_index = random_number_generator.permutation(range(len(X)))
        self.X = X[shuffled_index]
        self.y = y[shuffled_index]
        self.batch_size = batch_size
        self.batch_starting_indices = range(0, len(self.X), batch_size)

    @abstractmethod
    def __iter__(self):
        pass


class MiniBatchGenerator(BaseTrainingBatchGenerator):

    def __iter__(self):
        for start_index in self.batch_starting_indices:
            yield TrainingBatch(
                X=self.X[start_index : start_index + self.batch_size],
                y=self.y[start_index : start_index + self.batch_size]
            )
