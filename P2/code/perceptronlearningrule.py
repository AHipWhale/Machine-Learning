class PerceptronLearningRule:
    def __init__(self, weights, bias, learningrate):
        self.bias = bias
        self.weights = weights
        self.learningRate = learningrate
        self.errorList = []
        self.correct = 0
        self.target = 0

    def update(self, invoer, target):
        """Paste de learning rule toe zodat perceptron een bepaalde functie, zoals AND, kan leren"""
        # Deze if-statement kijkt of het nodig is om de weights en bias te veranderen, want als de output klopt is dat niet nodig
        if self.activation_function(invoer) == target:
            self.correct += 1
        else:
            self.target = target
            # berekent de error
            error = target - self.activation_function(invoer)

            for i in range(len(invoer)):
                # berekent de nieuwe weights
                w = self.learningRate * error * invoer[i]
                self.weights[i] = self.weights[i] + w

            # berekent delta bias
            b = self.learningRate * error

            # berekent de nieuwe bias
            self.bias = self.bias + b

            self.errorList.append(error)

    def error(self):
        """Deze functie returnt de MSE van deze perceptron"""
        return (sum(self.errorList)**2) / len(self.errorList)

    def activation_function(self, invoer: [float]):
        """Deze functie berekent de som van de inputs met we weights en telt daarbij de bias op"""
        self.invoer = invoer
        self.sumInvoer = 0
        # totale input word berekent
        for index in range(len(self.weights)):
            self.sumInvoer += invoer[index] * self.weights[index]
        self.sumInvoer += self.bias
        # sumInvoer word 0 of 1
        self.sumInvoer = self.output()

        return self.sumInvoer

    def output(self):
        """Deze functie kijkt of de som van de inputs*weights+bias hoger of lager is dan de 0 en returnt dan de output"""
        if self.sumInvoer >= 0:
            return 1
        else:
            return 0

    def __str__(self):
        """Deze functie returnt de belangrijke informatie van de perceptron.
        De bovenste return geeft de informatie wat makkelijker aan in notebook watn alles staat dan op 1 regel.
        De onderste return geeft de informatie wat netter weer."""

        return "Input: %s, weights: %s, bias: %s, target: %s, leraning rate: %s en dit gaf een output van %s" % \
                 (self.invoer, self.weights, self.bias, self.target, self.learningRate, self.sumInvoer)

        # return "De input was %s, de weights waren %s, een bias van %s, de target was %s, de leraning rate was %s en dit gaf een output van %s" % \
        #        (self.invoer, self.weights, self.bias, self.target, self.learningRate, self.sumInvoer)