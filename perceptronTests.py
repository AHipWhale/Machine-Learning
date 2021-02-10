import unittest
import perceptron as p
import perceptronlayer as pl
import perceptronnetwerk as pn

class MyTestCase(unittest.TestCase):

    def test_Invert(self):
        Not = p.Perceptron(weights=[-1], bias=0, trashhold=-0.5)

        antwoorden = []
        for verschillendeInputs in range(2):
            antwoorden.append(Not.output([verschillendeInputs]))

        self.assertEqual(antwoorden, [1, 0])

    def test_And(self):
        And = p.Perceptron(weights=[0.5, 0.5], bias=0, trashhold=1)

        antwoorden = []
        for verschillendeInputs1 in range(2):
            for verschillendeInputs2 in range(2):
                antwoorden.append(And.output([verschillendeInputs1, verschillendeInputs2]))

        self.assertEqual(antwoorden, [0, 0, 0, 1])

    def test_Or(self):
        Or = p.Perceptron(weights=[0.5, 0.5], bias=0, trashhold=0.5)

        antwoorden = []
        for verschillendeInputs1 in range(2):
            for verschillendeInputs2 in range(2):
                antwoorden.append(Or.output([verschillendeInputs1, verschillendeInputs2]))

        self.assertEqual(antwoorden, [0, 1, 1, 1])

    def test_Nor(self):
        Nor = p.Perceptron(weights=[-1, -1, -1], bias=0, trashhold=0)

        antwoorden = []
        for verschillendeInputs1 in range(2):
            for verschillendeInputs2 in range(2):
                for verschillendeInputs3 in range(2):
                    antwoorden.append(Nor.output([verschillendeInputs1, verschillendeInputs2, verschillendeInputs3]))

        self.assertEqual(antwoorden, [1, 0, 0, 0, 0, 0, 0, 0])\

    def test_Party(self):
        Party = p.Perceptron(weights=[0.6, 0.3, 0.2], bias=0, trashhold=0.4)

        antwoorden = []
        for verschillendeInputs1 in range(2):
            for verschillendeInputs2 in range(2):
                for verschillendeInputs3 in range(2):
                    antwoorden.append(Party.output([verschillendeInputs1, verschillendeInputs2, verschillendeInputs3]))

        self.assertEqual(antwoorden, [0, 0, 0, 1, 1, 1, 1, 1])

    def test_Xor(self):
        perceptron1 = p.Perceptron([1, 1], 0, 1)
        perceptron2 = p.Perceptron([-1, -1], 0, -1.5)

        layer1 = pl.PerceptronLayer([perceptron1, perceptron2])

        perceptron3 = p.Perceptron([1, 1], 0, 2)

        layer2 = pl.PerceptronLayer([perceptron3])

        Xor = pn.PerceptronNetwerk([layer1, layer2])

        antwoorden = []
        for verschillendeInputs1 in range(2):
            for verschillendeInputs2 in range(2):
                antwoorden.append(Xor.feed_forward([verschillendeInputs1, verschillendeInputs2]))

        self.assertEqual(antwoorden, [[0], [1], [1], [0]])

    def test_halfAdder(self):
        perceptron1 = p.Perceptron([1, 1], 0, 1)
        perceptron2 = p.Perceptron([-1, -1], 0, -1.5)
        perceptron3 = p.Perceptron([1, 1], 0, 2)

        layer1 = pl.PerceptronLayer([perceptron1, perceptron2, perceptron3])

        perceptron4 = p.Perceptron([1, 1, 0], 0, 2)
        perceptron5 = p.Perceptron([0, 0, 1], 0, 1)

        layer2 = pl.PerceptronLayer([perceptron4, perceptron5])

        halfAdder = pn.PerceptronNetwerk([layer1, layer2])

        antwoorden = []
        for verschillendeInputs1 in range(2):
            for verschillendeInputs2 in range(2):
                antwoorden.append(halfAdder.feed_forward([verschillendeInputs1, verschillendeInputs2]))

        # Elk antwoord bestaat uit 2 binaire getallen. Waar het eerste getal de sum is en de tweede de carry
        self.assertEqual(antwoorden, [[0, 0], [1, 0], [1, 0], [0, 1]])


if __name__ == '__main__':
    unittest.main()
