class NeuronLayer:
    def __init__(self, neuron):
        self.neuron = neuron
        self.input = []
        self.output = []

    def outputFunctie(self, input: list):
        """Deze functie geeft de outputs van alle neuronen in de layer"""
        outputs = []
        self.input = input
        for neuron in self.neuron:
            outputs.append(neuron.activation_function(input))
        self.output = outputs
        return outputs

    def __str__(self):
        """Deze functie returnt de belangrijke informatie van de neuron"""
        return "De lengte van deze layer is %s neuronen en de output van deze neuronen is %s" % \
               (len(self.neuron), self.outputFunctie(self.input))