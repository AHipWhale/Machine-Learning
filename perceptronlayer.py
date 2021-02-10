class PerceptronLayer:
    def __init__(self, perceptrons):
        self.perceptrons = perceptrons
        self.input = []
        self.output = []

    def outputFunctie(self, input: list):
        outputs = []
        self.input = input
        for perceptron in self.perceptrons:
            outputs.append(perceptron.output(input))
        self.output = outputs
        return outputs

    def __str__(self):
        """Deze functie returnt de belangrijke informatie van de perceptron"""
        return "De input was %s perceptrons en de output van deze perceptrons is %s" % \
               (len(self.perceptrons), self.outputFunctie(self.input))