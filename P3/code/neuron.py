import math

class Neuron:
    def __init__(self, weights, bias):
        self.bias = bias
        self.weights = weights
        self.sumInvoer = None

    def calculate_input(self, invoer: [float]):
        """Deze functie berekent de som van de inputs met we weights en telt daarbij de bias op"""
        self.sumInvoer = 0
        for index in range(len(self.weights)):
            self.sumInvoer += invoer[index] * self.weights[index]
        self.sumInvoer += self.bias
        return self.sumInvoer

    def activation_function(self, invoer: [float]):
        """Deze functie berekent de sigmoid en returnt dat de output"""
        return 1/(1+math.exp(-self.calculate_input(invoer)))

    def __str__(self):
        """Deze functie returnt de belangrijke informatie van de neuron"""
        return "De weights waren %s en een bias van %s. De output van de neuron was %s" % \
               (self.weights, self.bias, self.sumInvoer)