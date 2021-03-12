import unittest
from P4.code import neuron as n
from P4.code import neuronlayer as nl
from P4.code import neuronnetwork as nn

class MyTestCase(unittest.TestCase):
    def test_AND(self):
        inputs = [[0, 0],
                  [1, 0],
                  [0, 1],
                  [1, 1]]

        targets = [[0],
                   [0],
                   [0],
                   [1]]

        o = n.Neuron([-0.5, 0.5], 1.5, 1)

        output = nl.NeuronLayer([o])
        netwerk = nn.NeuronNetwerk([output])

        self.assertEqual(netwerk.train(inputs, targets), targets)

    def test_XOR(self):
        inputs = [[0, 0],
                  [1, 0],
                  [0, 1],
                  [1, 1]]

        targets = [[0],
                   [1],
                   [1],
                   [0]]

        f = n.Neuron([0.2, -0.4], 0, 1)
        g = n.Neuron([0.7, 0.1], 0, 1)
        o = n.Neuron([0.6, 0.9], 0, 1)

        hidden = nl.NeuronLayer([f, g])
        output = nl.NeuronLayer([o])
        netwerk = nn.NeuronNetwerk([hidden, output])

        self.assertEqual(netwerk.train(inputs, targets), targets)

    def test_ADDER(self):
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

        netwerk = nn.NeuronNetwerk([hidden, output])

        self.assertEqual(netwerk.train(inputs, targets), targets)

if __name__ == '__main__':
    unittest.main()
