import numpy as np
from lxmls.deep_learning.mlp import MLP
from lxmls.deep_learning.utils import index2onehot, logsumexp


class NumpyMLP(MLP):
    """
    Basic MLP with forward-pass and gradient computation in Numpy
    """

    def __init__(self, **config):

        # This will initialize
        # self.config
        # self.parameters
        MLP.__init__(self, **config)

    def predict(self, input=None):
        """
        Predict model outputs given input
        """
        log_class_probabilities, _ = self.log_forward(input)
        return np.argmax(np.exp(log_class_probabilities), axis=1)

    def update(self, input=None, output=None):
        """
        Update model parameters given batch of data
        """

        gradients = self.backpropagation(input, output)

        learning_rate = self.config['learning_rate']
        num_parameters = len(self.parameters)
        for m in np.arange(num_parameters):

            # Update weight
            self.parameters[m][0] -= learning_rate * gradients[m][0]

            # Update bias
            self.parameters[m][1] -= learning_rate * gradients[m][1]

    def log_forward(self, input):
        """Forward pass for sigmoid hidden layers and output softmax"""

        # Input
        tilde_z = input
        layer_inputs = []

        # Hidden layers
        num_hidden_layers = len(self.parameters) - 1
        for n in range(num_hidden_layers):

            # Store input to this layer (needed for backpropagation)
            layer_inputs.append(tilde_z)

            # Linear transformation
            weight, bias = self.parameters[n]
            z = np.dot(tilde_z, weight.T) + bias

            # Non-linear transformation (sigmoid)
            tilde_z = 1.0 / (1 + np.exp(-z))

        # Store input to this layer (needed for backpropagation)
        layer_inputs.append(tilde_z)

        # Output linear transformation
        weight, bias = self.parameters[num_hidden_layers]
        z = np.dot(tilde_z, weight.T) + bias

        # Softmax is computed in log-domain to prevent underflow/overflow
        log_tilde_z = z - logsumexp(z, axis=1, keepdims=True)

        return log_tilde_z, layer_inputs

    def cross_entropy_loss(self, input, output):
        """Cross entropy loss"""
        num_examples = input.shape[0]
        log_probability, _ = self.log_forward(input)
        return -log_probability[range(num_examples), output].mean()

    def backpropagation(self, input, output):
        """Gradients for sigmoid hidden layers and output softmax"""

        # Run forward and store activations for each layer
        log_prob_y, layer_inputs = self.log_forward(input)  ## predicted outputs
        prob_y = np.exp(log_prob_y)             ## move scores out of log domain

        num_examples, num_clases = prob_y.shape
        num_hidden_layers = len(self.parameters) - 1

        # For each layer in reverse store the backpropagated error, then compute
        # the gradients from the errors and the layer inputs
        errors = list()

        # ----------
        # Solution to Exercise 2

        ## compute error at last layer
        true_y = index2onehot(output, num_clases)
        err = true_y - prob_y
        # err = (prob_y - true_y) / num_examples

        ## ^ specific for Cross-Entropy loss
        ## (which gives a particularly friendly derivative)

        errors.append(err)

        ## backpropagate error through remaining layers
        for i in reversed(range(num_hidden_layers)):

            ## backpropagate through linear layer
            err = np.dot(err, self.parameters[i+1][0])

            ## backpropagate through the non-linearity (i.e. sigmoid layer)
            err *= layer_inputs[i+1] * (1 - layer_inputs[i+1])

            errors.append(err)

        ## reverse errors
        errors = errors[::-1]

        ## compute gradients from errors
        gradients = list()
        for i in range(num_hidden_layers + 1):
            w_i = self.parameters[i][0]
            w_grad = np.zeros(w_i.shape)

            ## weight gradient
            for l in range(num_examples):
                w_grad += np.outer(
                    errors[i][l, :],
                    layer_inputs[i][l, :]
                )

            ## bias gradient
            b_grad = np.sum(errors[i], axis=0, keepdims=True)

            ## line 19 of Algorithm 8 (SGD)
            ## (this could've be done at the beginning -- on the first error calculation,
            ##  as it propagates through following dot products)
            w_grad *= -1 / num_examples
            b_grad *= -1 / num_examples

            gradients.append([w_grad, b_grad])

        # End of solution to Exercise 2
        # ----------

        return gradients
