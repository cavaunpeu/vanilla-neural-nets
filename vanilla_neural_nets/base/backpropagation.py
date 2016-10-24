from abc import ABCMeta, abstractmethod


class BaseBackPropagateThroughTime(metaclass=ABCMeta):

    def __init__(self, feed_forward_method, backprop_through_time_steps, vocabulary_size, parameters):
        self.feed_forward_method = feed_forward_method
        self.backprop_through_time_steps = backprop_through_time_steps
        self.vocabulary_size = vocabulary_size
        self.parameters = parameters

    @abstractmethod
    def compute_gradients(self):
        pass
