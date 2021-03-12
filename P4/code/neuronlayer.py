class NeuronLayer:
    def __init__(self, neuron: list):
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

    def calc_deltas_layer(self, target, isOutputNeuron, weightsvorigelayer=None, errorvorigelayer=None):
        """Deze functie zorgt ervoor dat de juiste berekeningen word uitgevoerd en
           dat de juiste data word meegegeven aan de neuronen."""
        weightsPrefLayer = []
        errorPrefLayer = []

        if isOutputNeuron == True:
            # deze code is voor de output neuronen
            for neuron in self.neuron:
                neuron.calculate_error_outputNeuron(target[self.neuron.index(neuron)])
                neuron.calculate_deltaWeights()
                neuron.calculate_deltaBias()
                weightsPrefLayer.append(neuron.weights)
                errorPrefLayer.append(neuron.error)
            return weightsPrefLayer, errorPrefLayer

        else:
            # deze code is voor de hidden neuronen
            for i in range(len(self.neuron)):
                weights = [y[i] for y in weightsvorigelayer]
                errorPrefLayer.append(self.neuron[i].calculate_error_hiddenNeuron(weights, errorvorigelayer))
                self.neuron[i].calculate_deltaWeights()
                self.neuron[i].calculate_deltaBias()

                weightsPrefLayer.append(self.neuron[i].weights)
            return weightsPrefLayer, errorPrefLayer

    def __str__(self):
        """Deze functie returnt de belangrijke informatie van de neuron"""
        return "De lengte van deze layer is %s neuronen en de output van deze neuronen is %s" % \
               (len(self.neuron), self.outputFunctie(self.input))