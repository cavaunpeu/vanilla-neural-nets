from abc import ABCMeta, abstractmethod


class BaseRNNGradientDescent(metaclass=ABCMeta):

    def __init__(self, x, y, feed_forward_method, backprop_through_time_class,
        backprop_through_time_steps, learning_rate, vocabulary_size, parameters):
        self.x=x
        self.y=y
        self.feed_forward_method=feed_forward_method
        self.backprop_through_time_class = backprop_through_time_class
        self.backprop_through_time_steps=backprop_through_time_steps
        self.learning_rate=learning_rate
        self.vocabulary_size=vocabulary_size
        self.parameters=parameters

    def run(self):
        self._compute_gradients()
        self._update_parameters()
        return self.parameters

    def _compute_gradients(self):
        self.parameters = self.backprop_through_time_class(
            feed_forward_method=self.feed_forward_method,
            backprop_through_time_steps=self.backprop_through_time_steps,
            vocabulary_size=self.vocabulary_size,
            parameters=self.parameters
        ).compute_gradients(x=self.x, y=self.y)

    @abstractmethod
    def _update_parameters(self):
        pass
