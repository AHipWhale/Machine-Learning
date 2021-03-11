class NeuronNetwerk:
    def __init__(self, layers: list):
        self.layers = layers
        self.output = []
        self.netwerkError = 100000000

    def feed_forward(self, input: [float]):
        """Deze fuctie krijgt een input en haalt het door de layers heen.
            Elke layer veranderd de input naar de output van de vorige layer"""
        volgendeInput = input
        for layer in self.layers:
            volgendeInput = layer.outputFunctie(volgendeInput)
            # print(layer)
        self.output = volgendeInput
        return volgendeInput

    def updateNetwerk(self, invoer, target):
        self.feed_forward(invoer)
        self.loss(target)

        isOutputNeuron = True
        weights = None
        error = None

        for layer in reversed(self.layers):
            # print(layer, weights, error)
            # print(error)
            weights, error = layer.errors(target, isOutputNeuron, weights, error)

            isOutputNeuron = False

        for l in self.layers:
            for n in l.neuron:
                # print(n)
                n.update()
        # self.layers[-1].errors(invoer, target)

    def loss(self, targets):
        loss = 0
        outputs = self.layers[-1].output

        for i in range(len(outputs)):
            loss += 0.5 * (targets[i] - outputs[i])**2
        self.netwerkError = loss
        print(loss)

    def train(self, inputs: list, targets: list):
        #check of argumenten correct is
        if len(self.layers[0].neuron[0].weights) == len(inputs[0]) and len(self.layers[-1].neuron) == len(targets[0]):
            epoch = 0
            while epoch <= 1000000 and self.netwerkError >= 0.0001:
                for i in range(len(inputs)):
                    self.updateNetwerk(inputs[i], targets[i])
                    self.loss(targets[i])
                epoch += 1
            print(epoch)

            for input in inputs:
                self.feed_forward(input)
                print([round(i) for i in self.output])
        else:
            print("De meegegeven argumenten komen niet overeen met het aantal nodige inputs of targets")

    def __str__(self):
        """Deze functie returnt de belangrijke informatie van het neuron netwerk"""
        return "Dit netwerk heeft %s layers en de output van dit netwerk is %s" % (self.layers, self.output)

import neuron as n
import neuronlayer as nl

# f = n.Neuron([0.2, -0.4], 0, 1)
# g = n.Neuron([0.7, 0.1], 0, 1)
# o = n.Neuron([0.6, 0.9], 0, 1)
#
# hidden = nl.NeuronLayer([f, g])
# output = nl.NeuronLayer([o])
# netwerk = NeuronNetwerk([hidden, output])
#
# print("iteratie 1")
# netwerk.updateNetwerk([1, 1], [0])
#
# print("iteratie 2")
# netwerk.updateNetwerk([0, 1], [1])

# o = n.Neuron([-0.5, 0.5], 1.5 , 1)
#
# output = nl.NeuronLayer([o])
# netwerk = NeuronNetwerk([output])
#
# print("iteratie 1")
# netwerk.updateNetwerk([0, 0], [0])
#
# print("iteratie 2")
# netwerk.updateNetwerk([1, 0], [0])
#
# print("iteratie 3")
# netwerk.updateNetwerk([0, 1], [0])
#
# print("iteratie 4")
# netwerk.updateNetwerk([1, 1], [1])

# f = n.Neuron([0, 0.1], 0, 1)
# g = n.Neuron([0.2, 0.3], 0, 1)
# h = n.Neuron([0.4, 0.5], 0, 1)
#
# s = n.Neuron([0.6, 0.7, 0.8], 0, 1)
# c = n.Neuron([0.9, 1, 1.1], 0, 1)
#
# hidden = nl.NeuronLayer([f, g, h])
# output = nl.NeuronLayer([s, c])
#
# netwerk = NeuronNetwerk([hidden, output])
#
# print("iteratie 1")
# netwerk.updateNetwerk([1, 1], [0, 1])

# netwerk.train([1, 1], [0, 1])

inputs = [[0, 0],
          [1, 0],
          [0, 1],
          [1, 1]]

targets = [[0, 0],
           [1, 0],
           [1, 0],
           [0, 1]]

f = n.Neuron([0, 0.1], 0, 1)
g = n.Neuron([0.2, 0.3], 0, 1)
h = n.Neuron([0.4, 0.5], 0, 1)

s = n.Neuron([0.6, 0.7, 0.8], 0, 1)
c = n.Neuron([0.9, 1, 1.1], 0, 1)

hidden = nl.NeuronLayer([f, g, h])
output = nl.NeuronLayer([s, c])

netwerk = NeuronNetwerk([hidden, output])

netwerk.train(inputs, targets)