import math

class Neuron:
    def __init__(self, weights, bias, learning_rate):
        self.bias = bias
        self.weights = weights
        self.sumInvoer = None
        self.aanpassingWeights = []
        self.learning_rate = learning_rate

    def calculate_input(self, invoer: [float]):
        """Deze functie berekent de som van de inputs met we weights en telt daarbij de bias op"""
        self.sumInvoer = 0
        for index in range(len(self.weights)):
            self.sumInvoer += invoer[index] * self.weights[index]
        self.sumInvoer += self.bias
        return self.sumInvoer

    def activation_function(self, invoer: [float]):
        """Deze functie berekent de sigmoid en returnt dat als output"""
        return 1/(1+math.exp(-self.calculate_input(invoer)))

    def calculate_error(self, invoer: [float], target):
        output = self.activation_function(invoer)
        afgeleide = output * (1 - output)
        return afgeleide * -(target - output)

    def calculate_gradient(self, output, invoer: [float], target):
        return output * self.calculate_error(invoer, target)

    def calculate_deltaWeights(self,invoer: [float], target):
        for i in range(len(self.weights)):
            self.aanpassingWeights.append(self.learning_rate * self.calculate_error(invoer, target) * invoer[i])

    def calculate_deltaBias(self, invoer: [float], target):
        return self.learning_rate * self.calculate_error(invoer, target)

    def update(self, invoer: [float], target):
        self.calculate_deltaWeights(invoer, target)
        for i in range(len(self.aanpassingWeights)):
            self.weights[i] -= self.aanpassingWeights[i]

        self.bias -= self.calculate_deltaBias(invoer, target)

    def __str__(self):
        """Deze functie returnt de belangrijke informatie van de neuron"""
        return "De weights waren %s en een bias van %s. De output van de neuron was %s" % \
               (self.weights, self.bias, self.sumInvoer)

n1 = Neuron([-0.5, 0.5], 1.5, 1)

n1.update([0,0], 0)

print(n1)