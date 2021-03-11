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
        # print('output', self.output)
        afgeleide = self.output * (1 - self.output)
        # print("afgeleide", afgeleide)
        self.error = afgeleide * -(target - self.output)
        # print("error", self.error)
        return self.error

    def calculate_error_hiddenNeuron(self, volgendeWeights, volgendeError):
        sumError = 0
        # print("nextWeights", volgendeWeights)
        # print("nextError", volgendeError)
        for i in range(len(volgendeWeights)):
            sumError += volgendeWeights[i] * volgendeError[i]
        # print("sumError", sumError)
        # print("output", self.output)
        afgeleide = self.output * (1 - self.output)
        self.error = afgeleide * sumError
        # print("error", self.error)
        return self.error

    def calculate_gradient(self, delta):
        return self.output * delta

    def calculate_deltaWeights(self):
        for i in range(len(self.weights)):
            self.aanpassingWeights.append(self.learning_rate * self.invoer[i] * self.error)

    def calculate_deltaBias(self):
        self.aanpassingBias = self.learning_rate * self.error
        return self.aanpassingBias

    def update(self):
        self.aanpassingWeights = []
        # print("NEURON", self.__str__())
        self.calculate_deltaWeights()
        # print("\ndeltaWeights", self.aanpassingWeights)
        for i in range(len(self.aanpassingWeights)):
            # print(self.weights[i], self.aanpassingWeights[i])
            self.weights[i] -= self.aanpassingWeights[i]
            #self.weights[i] -= learning_rate * self.aanpassingWeights[i]
            # print("newWeights", self.weights[i])
        self.bias -= self.calculate_deltaBias()
        # print("newBias", self.bias, "\n")

    def __str__(self):
        """Deze functie returnt de belangrijke informatie van de neuron"""
        return "De weights waren %s en een bias van %s. De output van de neuron was %s" % \
               (self.weights, self.bias, self.output)