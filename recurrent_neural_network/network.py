
class VanillaRecurrentNeuralNetwork:
    
    # you should pass in a _feed_forward function from inside of your network to the optimization routine, so you don't repeat yourself.
    
    def __init__(self, vocabulary_size, hidden_layer_size, backprop_through_time_steps, 
        optimization_algorithm_class, learning_rate, n_epochs, parameter_initializer, random_state):
        self.vocabulary_size = vocabulary_size
        self.hidden_layer_size = hidden_layer_size
        self.backprop_through_time_steps = backprop_through_time_steps
        self.optimization_algorithm_class = optimization_algorithm_class
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.parameter_initializer = parameter_initializer
        self.random_state = random_state
