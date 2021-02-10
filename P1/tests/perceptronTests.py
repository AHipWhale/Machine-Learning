import unittest
from P1.code import perceptron as p, perceptronlayer as pl, perceptronnetwerk as pn


class MyTestCase(unittest.TestCase):

    def test_Invert(self):
        """Hier word de NOT getest met 1 input"""

        # Perceptron word aangemaakt
        Not = p.Perceptron(weights=[-1], bias=0, trashhold=-0.5)

        antwoorden = []
        # Alle mogelijkheden worden getest op de perceptron
        for verschillendeInputs in range(2):
            antwoorden.append(Not.output([verschillendeInputs]))

        self.assertEqual(antwoorden, [1, 0])

    def test_And(self):
        """Hier word de AND getest met 2 inputs"""

        And = p.Perceptron(weights=[0.5, 0.5], bias=0, trashhold=1)

        antwoorden = []
        for verschillendeInputs1 in range(2):
            for verschillendeInputs2 in range(2):
                antwoorden.append(And.output([verschillendeInputs1, verschillendeInputs2]))

        self.assertEqual(antwoorden, [0, 0, 0, 1])

    def test_Or(self):
        """Hier word de OR getest met 2 inputs"""

        Or = p.Perceptron(weights=[0.5, 0.5], bias=0, trashhold=0.5)

        antwoorden = []
        for verschillendeInputs1 in range(2):
            for verschillendeInputs2 in range(2):
                antwoorden.append(Or.output([verschillendeInputs1, verschillendeInputs2]))

        self.assertEqual(antwoorden, [0, 1, 1, 1])

    def test_Nor(self):
        """Hier word de NOR getest met 3 inputs met 3 inputs"""

        Nor = p.Perceptron(weights=[-1, -1, -1], bias=0, trashhold=0)

        antwoorden = []
        for verschillendeInputs1 in range(2):
            for verschillendeInputs2 in range(2):
                for verschillendeInputs3 in range(2):
                    antwoorden.append(Nor.output([verschillendeInputs1, verschillendeInputs2, verschillendeInputs3]))

        self.assertEqual(antwoorden, [1, 0, 0, 0, 0, 0, 0, 0])\

    def test_Party(self):
        """Hier word de party uit de reader getest met 3 inputs"""

        Party = p.Perceptron(weights=[0.6, 0.3, 0.2], bias=0, trashhold=0.4)

        antwoorden = []
        for verschillendeInputs1 in range(2):
            for verschillendeInputs2 in range(2):
                for verschillendeInputs3 in range(2):
                    antwoorden.append(Party.output([verschillendeInputs1, verschillendeInputs2, verschillendeInputs3]))

        self.assertEqual(antwoorden, [0, 0, 0, 1, 1, 1, 1, 1])

    def test_Xor(self):
        """Hier word de XOR getest met 2 layers en in totaal 3 perceptrons"""

        # Perceptrons worden aangemaakt
        perceptron1 = p.Perceptron([1, 1], 0, 1)
        perceptron2 = p.Perceptron([-1, -1], 0, -1.5)

        # Layer word aangemaakt en perceptrons worden toegevoegd aan layer
        layer1 = pl.PerceptronLayer([perceptron1, perceptron2])

        # Perceptron worden aangemaakt
        perceptron3 = p.Perceptron([1, 1], 0, 2)

        # Layer word aangemaakt en percepotron word toegevoegd aan layer
        layer2 = pl.PerceptronLayer([perceptron3])

        # Netwerk word aangemaakt en layers worden toegevoegd aan netwerk
        Xor = pn.PerceptronNetwerk([layer1, layer2])

        antwoorden = []

        for verschillendeInputs1 in range(2):
            for verschillendeInputs2 in range(2):
                antwoorden.append(Xor.feed_forward([verschillendeInputs1, verschillendeInputs2]))

        self.assertEqual(antwoorden, [[0], [1], [1], [0]])

    def test_halfAdder(self):
        """Hier word de half Adder getest met 2 layers en in totaal 5 perceptrons
            (Hier word vrijwel hetzelfde gedaan als bij de XOR maar met 2 extra perceptrons)"""

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
