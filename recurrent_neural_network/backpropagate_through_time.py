import numpy as np


class _BackPropagateThroughTime:

    def __init__(self, feed_forward_method, backprop_through_time_steps, vocabulary_size, parameters):
        self.feed_forward_method = feed_forward_method
        self.backprop_through_time_steps = backprop_through_time_steps
        self.vocabulary_size = vocabulary_size
        self.parameters = parameters

    def compute_gradients(self, x, y):
        softmax_outputs, hidden_state = self.feed_forward_method(x)
        self.parameters.reset_gradients_to_zero()
        time_steps = np.arange(len(x))

        for t in time_steps[::-1]:
            label = y[t]

            # derivative of loss function w.r.t. softmax predictions
            dJdP = softmax_outputs[t]
            dJdP[label] -= 1

            # derivative of loss function w.r.t. W_hy
            self.parameters.W_hy.gradient += np.outer(dJdP, hidden_state[t])

            # derivative of loss function w.r.t. hidden state
            dJdH = dJdP @ self.parameters.W_hy.value

            # initialize dJdH_parent
            dJdH_parent = dJdH

            # back-propagate through time
            back_prop_through_time_steps = np.arange( max(0, t-self.backprop_through_time_steps), t+1)
            for t_ in back_prop_through_time_steps[::-1]:

                # derivative of loss function w.r.t. hidden-layer input
                dJdZ = dJdH_parent * (1 - hidden_state[t_]**2)

                # derivative of loss function w.r.t. W_hh
                self.parameters.W_hh.gradient += np.outer(dJdZ, hidden_state[t_-1])

                # derivative of loss function w.r.t. W_xh
                input_vector = np.arange(self.vocabulary_size) == x[t_]
                self.parameters.W_xh.gradient += np.outer(dJdZ, input_vector)

                # derivative of loss function w.r.t. *previous* hidden state
                dJdH_parent = dJdZ @ self.parameters.W_hh.value

        return self.parameters
