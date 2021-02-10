class Perceptron:
    def __init__(self, weights, bias, trashhold):
        self.bias = bias
        self.weights = weights
        self.trashhold = trashhold
        self.outputs = []

    def activation_function(self, invoer: [float]):
        """Deze functie berekent de som van de inputs met we weights en telt daarbij de bias op"""
        sumInvoer = 0
        for index in range(len(self.weights)):
            sumInvoer += invoer[index] * self.weights[index]
        sumInvoer += self.bias
        return sumInvoer

    def output(self, input: [float]):
        """Deze functie kijkt of de som van de inputs*weights+bias of lager is dan de treshold en returnt dan de output"""
        if self.activation_function(input) >= self.trashhold:
            self.outputs.append(1)
            return 1
        else:
            self.outputs.append(0)
            return 0

    def __str__(self):
        """Deze functie returnt de belangrijke informatie van de perceptron"""
        return "De weights waren %s en een bias van %s. De trashhold was %s en dit gaf een output van %s" % \
               (self.weights, self.bias, self.trashhold, self.outputs)