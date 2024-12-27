import math
import random


def parse_image():
    pass


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


class MLP:
    def __init__(
        self,
        activation_function,
        activation_derivative,
        hidden_layers,
        hidden_units,
        input_units,
        output_units,
        learning_rate,
        training_set,
        initial_weights=None,
    ):
        self.activation_function = activation_function
        self.activation_derivative = activation_derivative

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

    def softmax(self, z):
        total = sum([math.exp(z_j) for z_j in z])
        return [math.exp(z_i) / total for z_i in z]

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
        err = self.sub(y, h_w)  # err[i] is the individual error of output neuron i
        L = len(self.weights) - 1

        deltas = [
            err[i]
            * self.activation_derivative(
                self.dot(a[L - 1], self.get_incoming_weights(L, i))
            )
            for i in self.output_units
        ]

        # Backpropagate
        for l in range(L, 0, -1):
            new_deltas = []

            if l > 0:
                prev_unit_count = self.hidden_units if l > 1 else self.output_units

                for i in range(prev_unit_count):
                    curr_weights = self.weights[l][i]

                    value = self.dot(a[L - 1], self.get_incoming_weights(L, i))
                    value_dev = self.activation_derivative(value)
                    prop_delta = sum(
                        [curr_weights[j] * deltas[j] for j in range(len(curr_weights))]
                    )

                    new_deltas.append(prop_delta * value_dev)

            # Update weights using gradient descent
            for i, neuron_weights in enumerate(self.weights[l]):
                self.weights[l][i] = [
                    w + self.learning_rate * deltas[j] * self.values[l][i]
                    for w, j in enumerate(neuron_weights[i])
                ]

            if l > 0:
                deltas = new_deltas

    def train(self, epochs):
        for _ in epochs:
            for sample in self.training_set:
                self.backpropagate(sample[0], sample[1])

    def test(self, x, y):
        outputs, _ = self.forward_propagate(x)
        return self.sub(y, outputs)

    def evaluate(self, x):
        outputs, _ = self.forward_propagate(x)
        return self.softmax(outputs)


def run(image):
    pass


if __name__ == "__main__":
    pass
