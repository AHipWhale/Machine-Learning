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

    def errors(self, target, isOutputNeuron, weightsvorigelayer=None, errorvorigelayer=None):
        weightsPrefLayer = []
        errorPrefLayer = []
        errorOfLayer = []

        if isOutputNeuron == True:
            for neuron in self.neuron:
                # print(neuron)
                output = neuron.output
                # print(output, target[self.neuron.index(neuron)])
                neuron.calculate_error_outputNeuron(target[self.neuron.index(neuron)])
                neuron.calculate_deltaWeights()
                neuron.calculate_deltaBias()
                weightsPrefLayer.append(neuron.weights)
                errorPrefLayer.append(neuron.error)
                # print("error outputlayer", errorPrefLayer)
            return weightsPrefLayer, errorPrefLayer

        else:
            for i in range(len(self.neuron)):
                # print("neuron",self.neuron[i])
                # output = self.neuron[i].activation_function(invoer)
                # print("weightsvorigelayer", weightsvorigelayer)
                # print("index", i)
                weights = [y[i] for y in weightsvorigelayer]
                # print("weights", weights)
                # print("errorPrefLayer", errorvorigelayer)
                errorOfLayer.append(self.neuron[i].calculate_error_hiddenNeuron(weights, errorvorigelayer))
                self.neuron[i].calculate_deltaWeights()
                self.neuron[i].calculate_deltaBias()

                weightsPrefLayer.append(self.neuron[i].weights)
            return weightsPrefLayer, errorOfLayer

    def __str__(self):
        """Deze functie returnt de belangrijke informatie van de neuron"""
        return "De lengte van deze layer is %s neuronen en de output van deze neuronen is %s" % \
               (len(self.neuron), self.outputFunctie(self.input))