from recurrent_neural_network.backpropagate_through_time import BackPropagateThroughTime


class RNNGradientDescent:

    def __init__(self, x, y, feed_forward_method, learning_rate, backprop_through_time_steps,
        vocabulary_size, parameters):
            self.x=x
            self.y=y
            self.feed_forward_method=feed_forward_method
            self.learning_rate=learning_rate
            self.backprop_through_time_steps=backprop_through_time_steps
            self.vocabulary_size=vocabulary_size
            self.parameters=parameters

    def run(self):
        self._compute_gradients()
        self._update_weights()
        return self.parameters

    def _compute_gradients(self):
        self.parameters = BackPropagateThroughTime(
            feed_forward_method=self.feed_forward_method,
            backprop_through_time_steps=self.backprop_through_time_steps,
            vocabulary_size=self.vocabulary_size,
            parameters=self.parameters
        ).compute_gradients(x=self.x, y=self.y)

    def _update_weights(self):
        self.parameters.W_xh.value -= self.learning_rate * self.parameters.W_xh.gradient
        self.parameters.W_hh.value -= self.learning_rate * self.parameters.W_hh.gradient
        self.parameters.W_hy.value -= self.learning_rate * self.parameters.W_hy.gradient
