from collections import deque

import numpy as np

from base.network import BaseRecurrentNeuralNetwork


class VanillaRecurrentNeuralNetwork(BaseRecurrentNeuralNetwork):

    def _feed_forward(self, x):
        time_steps = len(x)
        initial_hidden_state = np.zeros(self.hidden_layer_size)
        hidden_state = deque([initial_hidden_state])
        softmax_outputs = deque()

        for t in np.arange(time_steps):
            hidden_state.append(
                np.tanh( self.parameters.W_xh.value[:, x[t]] + self.parameters.W_hh.value @ hidden_state[-1] )
            )
            softmax_outputs.append(
                self._compute_softmax( self.parameters.W_hy.value @ hidden_state[-1] )
            )
        # move initial hidden state to end of deque, such that it is later our
        # `hidden_state[t-1]` at t=0
        hidden_state.rotate(-1)

        return np.array(softmax_outputs), np.array(hidden_state)
