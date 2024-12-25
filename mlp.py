import math
import random

import numpy as np


def parse_image():
    pass


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def softmax():
    return


class MLP:
    def __init__(
        self,
        activation_function,
        hidden_layers,
        hidden_units,
        input_units,
        output_units,
        learning_rate,
        training_set,
        initial_weights=None,
    ):
        self.activation_function = activation_function

        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.input_units = input_units
        self.output_units = output_units

        self.learning_rate = learning_rate
        self.training_set = training_set

        # self.values = []  # values[k][j] is the value of neuron j in layer k
        self.weights = (
            initial_weights  # weights[k][j][i] is weight i from neuron j in layer k
        )

        # Generate hidden_layers + 2 layers of random weights
        if not initial_weights:
            self.weights = [
                [[random.random() for _ in hidden_units] for _ in input_units]
            ]

            for _ in range(hidden_layers - 1):
                self.weights.append(
                    [[random.random() for _ in hidden_units] for _ in hidden_units]
                )

            penultimate_neuron_count = (
                hidden_units if hidden_layers > 0 else output_units
            )

            self.weights.append(
                [
                    [random.random() for _ in output_units]
                    for _ in penultimate_neuron_count
                ]
            )

    def get_incoming_weights(self, layer, index):
        weights = []

        for neuron_weights in self.weights[layer]:
            weights.append(neuron_weights[index])

        return weights

    def dot(self, x1, x2):
        return sum([x1[i] * x2[i] for i in range(min(len(x1), len(x2)))])

    def sub(self, x1, x2):
        return [x1[i] - x2[i] for i in min(len(x1), len(x2))]

    def forward_propagate(self, x):
        h_t = x
        values = []

        for l, weights_l in enumerate(self.weights):
            for i in range(len(weights_l[0])):
                w = self.get_incoming_weights(l, i)
                value = self.activation_function(self.dot(w, h_t))

                h_t.append(value)

            values.append(h_t)

        return h_t, values

    def backpropagate(self, x, y):
        h_w, a = self.forward_propagate(x)
        err = self.sub(h_w, y)  # err[i] is the individual error of output neuron i

        deltas = []

        pass

    def train(self, epochs):
        pass

    def evaluate(self):
        pass


def run(image):
    pass


if __name__ == "__main__":
    pass
