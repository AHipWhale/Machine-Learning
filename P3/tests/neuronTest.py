import unittest
from P3.code import neuron as n
from P3.code import neuronlayer as nl
from P3.code import neuronnetwork as nn

class MyTestCase(unittest.TestCase):
    """
    Bij al deze test waar ik dezelfde parameters heb gebruikt als bij de perceptron. Heb ik niet de bias overgenomen
    maar -threshold gebruikt als bias.
    """

    def test_Invert(self):
        """Hier word de NOT getest met 1 input"""

        # Neuron word aangemaakt
        Not = n.Neuron(weights=[-1], bias=0.5)

        antwoorden = []
        # Alle mogelijkheden worden getest op de Neuron
        for verschillendeInputs in range(2):
            antwoorden.append(round(Not.activation_function([verschillendeInputs])))
        self.assertNotEqual(antwoorden, [1, 0])

        """Zoals je kan zien is de output van de neuron met dezelfde parameters als bij de perceptron hetzelfde.   
           Dit komt omdat de neuron met input 0 meer richting de 1 leunt en de neuron met input 0 leunt meet naar 0."""

    def test_And(self):
        """Hier word de AND getest met 2 inputs"""

        And = n.Neuron(weights=[0.5, 0.5], bias=-1)

        antwoorden = []
        for verschillendeInputs1 in range(2):
            for verschillendeInputs2 in range(2):
                antwoorden.append(round(And.activation_function([verschillendeInputs1, verschillendeInputs2])))

        self.assertNotEqual(antwoorden, [0, 0, 0, 1])

        """Met deze parameters is de output net niet goed. De output van de neuron waar de input [1, 1] is, is de output 0.5.
           Als je de afrond word het 0. Dit kan je oplossen door de bias wat te verhogen, bijvoorbeeld de bias -0.75 maken, 
           maar dit zou wellicht nog wat lager kunnen. Dit zorgt ervoor dat de outputs wat hoger worden. 
           De output van [1, 1] leunt nu meer naar de 1 toe maar de rest van de outputs nog naar de 0."""

        And = n.Neuron(weights=[0.5, 0.5], bias=-0.75)

        antwoorden = []
        for verschillendeInputs1 in range(2):
            for verschillendeInputs2 in range(2):
                antwoorden.append(round(And.activation_function([verschillendeInputs1, verschillendeInputs2])))

        self.assertEqual(antwoorden, [0, 0, 0, 1])

    def test_Or(self):
        """Hier word de OR getest met 2 inputs"""

        Or = n.Neuron(weights=[0.5, 0.5], bias=-0.5)

        antwoorden = []
        for verschillendeInputs1 in range(2):
            for verschillendeInputs2 in range(2):
                antwoorden.append(round(Or.activation_function([verschillendeInputs1, verschillendeInputs2])))

        self.assertNotEqual(antwoorden, [0, 1, 1, 1])

        """Hetzelfde als bij de AND geldt hier voor de OR. De middelste twee outputs zitten net op de helft. 
           Als je de bias dus een klein beetje verhoogt klopt de neuron wel."""

        Or = n.Neuron(weights=[0.5, 0.5], bias=-0.25)

        antwoorden = []
        for verschillendeInputs1 in range(2):
            for verschillendeInputs2 in range(2):
                antwoorden.append(round(Or.activation_function([verschillendeInputs1, verschillendeInputs2])))

        self.assertEqual(antwoorden, [0, 1, 1, 1])

    def test_Nor(self):
        """Hier word de NOR getest met 3 inputs"""

        Nor = n.Neuron(weights=[-1, -1, -1], bias=0)
        antwoorden = []
        for verschillendeInputs1 in range(2):
            for verschillendeInputs2 in range(2):
                for verschillendeInputs3 in range(2):
                    antwoorden.append(round(Nor.activation_function([verschillendeInputs1, verschillendeInputs2, verschillendeInputs3])))

        self.assertNotEqual(antwoorden, [1, 0, 0, 0, 0, 0, 0, 0])

        """Hetzelfde geldt bij de NOR ook. De eerste output (die 1 moet geven) zit precies in het midden. 
        Je zou dit dus weer kunnen verbeteren door de bias hoger te maken. 
        Zoals hieronder te zien is de bias maar heel weinig veranderd en toch geeft de neuron de goeie outputs."""

        Nor = n.Neuron(weights=[-1, -1, -1], bias=0.1)
        antwoorden = []
        for verschillendeInputs1 in range(2):
            for verschillendeInputs2 in range(2):
                for verschillendeInputs3 in range(2):
                    antwoorden.append(round(
                        Nor.activation_function([verschillendeInputs1, verschillendeInputs2, verschillendeInputs3])))

        self.assertEqual(antwoorden, [1, 0, 0, 0, 0, 0, 0, 0])

    def test_halfAdder(self):
        """Hier word de half Adder getest met 2 layers en in totaal 5 perceptrons
            (Hier word vrijwel hetzelfde gedaan als bij de XOR maar met 2 extra perceptrons)"""

        perceptron1 = n.Neuron([1, 1], 0)
        perceptron2 = n.Neuron([0, -1], 0)
        perceptron3 = n.Neuron([0, 0], 0)

        layer1 = nl.NeuronLayer([perceptron1, perceptron2, perceptron3])

        perceptron4 = n.Neuron([1, -1, 0], 0)
        perceptron5 = n.Neuron([-1, 0.5, 1.5], 0)

        layer2 = nl.NeuronLayer([perceptron4, perceptron5])

        halfAdder = nn.NeuronNetwerk([layer1, layer2])

        antwoorden = []
        for verschillendeInputs1 in range(2):
            for verschillendeInputs2 in range(2):
                antwoorden.append(halfAdder.feed_forward([verschillendeInputs1, verschillendeInputs2]))

        # Elk antwoord bestaat uit 2 binaire getallen. Waar het eerste getal de sum is en de tweede de carry
        self.assertEqual(antwoorden, [[0, 0], [1, 0], [1, 0], [0, 1]])

if __name__ == '__main__':
    unittest.main()
