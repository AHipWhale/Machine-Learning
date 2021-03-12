import math

class Neuron:
    def __init__(self, weights, bias, learning_rate):
        self.bias = bias
        self.weights = weights
        self.sumInvoer = None
        self.aanpassingWeights = []
        self.aanpassingBias = None
        self.learning_rate = learning_rate
        self.error = None
        self.output = None
        self.invoer = None

    def calculate_input(self, invoer: [float]):
        """Deze functie berekent de som van de inputs met we weights en telt daarbij de bias op"""
        self.sumInvoer = 0
        self.invoer = invoer
        for index in range(len(self.weights)):
            self.sumInvoer += invoer[index] * self.weights[index]
        self.sumInvoer += self.bias
        return self.sumInvoer

    def activation_function(self, invoer: [float]):
        """Deze functie berekent de sigmoid en returnt dat als output"""
        self.output = 1/(1+math.exp(-self.calculate_input(invoer)))
        return self.output

    def calculate_error_outputNeuron(self, target):
        """Deze functie berekent de error van de neuronen in de output layer"""
        afgeleide = self.output * (1 - self.output)
        self.error = afgeleide * -(target - self.output)
        return self.error

    def calculate_error_hiddenNeuron(self, volgendeWeights, volgendeError):
        """Deze functie berkent de error van de neuronen in de hidden layer"""
        sumError = 0
        for i in range(len(volgendeWeights)):
            sumError += volgendeWeights[i] * volgendeError[i]
        afgeleide = self.output * (1 - self.output)
        self.error = afgeleide * sumError
        return self.error

    def calculate_gradient(self, delta):
        """Deze functie berekent de gradient van de neuron"""
        return self.output * delta

    def calculate_deltaWeights(self):
        """Deze functie berekent de delta's van de weights van de neuron"""
        for i in range(len(self.weights)):
            self.aanpassingWeights.append(self.learning_rate * self.invoer[i] * self.error)

    def calculate_deltaBias(self):
        """Deze functie berekent de delta van de bias van de neuron"""
        self.aanpassingBias = self.learning_rate * self.error
        return self.aanpassingBias

    def update(self):
        """Deze functie berekent de nieuwe weights en bias van de neuron"""
        self.aanpassingWeights = []
        self.calculate_deltaWeights()
        for i in range(len(self.aanpassingWeights)):
            self.weights[i] -= self.aanpassingWeights[i]
        self.bias -= self.calculate_deltaBias()

    def __str__(self):
        """Deze functie returnt de belangrijke informatie van de neuron"""
        return "De weights waren %s en een bias van %s. De output van de neuron was %s" % \
               (self.weights, self.bias, self.output)