from abc import ABCMeta, abstractmethod


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
