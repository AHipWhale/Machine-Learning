class Perceptron:
    def __init__(self, invoer, weights, bias, trashhold):
        self.invoer = invoer
        self.bias = bias
        self.weights = weights
        self.trashhold = trashhold
        self.sumInvoer = 0

    def activation_function(self):
        # Deze functie berekent de som van de inputs met we weights en telt daarbij de bias op
        for index in range(len(self.weights)):
            self.sumInvoer += self.invoer[index] * self.weights[index]
        self.sumInvoer += self.bias
        return self.sumInvoer

    def output(self):
        # Deze functie kijkt of de som van de inputs hoger of lager is dan de treshold en returnt dan de output
        if self.activation_function() >= self.trashhold:
            return 1
        else:
            return 0

    def __str__(self):
        # Deze functie returnt de belangrijke informatie van de perceptron
        return "De input was %s met de weights %s en een bias van %s. De trashhold was %s en dit gaf een output van %s" % \
               (self.invoer, self.weights, self.bias, self.trashhold, self.output())