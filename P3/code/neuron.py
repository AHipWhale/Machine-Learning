import math

class Neuron:
    def __init__(self, weights, bias):
        self.bias = bias
        self.weights = weights
        self.outputs = []
        self.sumInvoer = None

    def activation_function(self, invoer: [float]):
        """Deze functie berekent de som van de inputs met we weights en telt daarbij de bias op"""
        self.sumInvoer = 0
        for index in range(len(self.weights)):
            self.sumInvoer += invoer[index] * self.weights[index]
        self.sumInvoer += self.bias
        return self.sumInvoer

    def output(self):
        """Deze functie kijkt of de som van de inputs*weights+bias of lager is dan de treshold en returnt dan de output"""
        return 1/(1+math.exp(-self.sumInvoer))

    def __str__(self):
        """Deze functie returnt de belangrijke informatie van de neuron"""
        return "De weights waren %s en een bias van %s. De output van de neuron was %s" % \
               (self.weights, self.bias, self.outputs)