import time

class NeuronNetwerk:
    def __init__(self, layers: list):
        self.layers = layers
        self.output = []
        self.loss = 100000000
        self.epoch = 0

    def feed_forward(self, input: [float]):
        """Deze fuctie krijgt een input en haalt het door de layers heen.
            Elke layer veranderd de input naar de output van de vorige layer"""
        volgendeInput = input
        for layer in self.layers:
            volgendeInput = layer.outputFunctie(volgendeInput)
        self.output = volgendeInput
        return volgendeInput

    def backpropagation(self, invoer, target):
        """Deze functie voert de backpropagation uit op het netwerk"""
        self.feed_forward(invoer)

        isOutputNeuron = True
        weights = None
        error = None

        # berekent de delta's van alle neuronen
        for layer in reversed(self.layers):
            weights, error = layer.calc_deltas_layer(target, isOutputNeuron, weights, error)

            isOutputNeuron = False

        # update alle neurons
        for l in self.layers:
            for n in l.neuron:
                n.update()

    def calc_loss(self, targets):
        """Deze functie berekent de loss van het netwerk."""
        loss = 0
        outputs = self.layers[-1].output

        for i in range(len(outputs)):
            loss += (targets[i] - outputs[i])**2
        self.loss = loss / len(outputs)

    def train(self, inputs: list, targets: list):
        """Deze functie traint het neuronnetwerk met behulp van meegegeven inputs en targets."""
        # checkt of argumenten correct zijn
        if len(self.layers[0].neuron[0].weights) == len(inputs[0]) and len(self.layers[-1].neuron) == len(targets[0]):
            start = time.time()
            self.epoch = 0
            # blijft trainen totdat max. epoch of loss onder grens ligt of de train() al 3 min runt
            while self.epoch <= 1000000 and self.loss >= 0.001 and (time.time() - start) <= 240:
                for i in range(len(inputs)):
                    # voert backpropagation uit
                    self.backpropagation(inputs[i], targets[i])
                    self.calc_loss(targets[i])
                self.epoch += 1

        else:
            # bericht als argumenten niet kloppen
            print("De meegegeven argumenten komen niet overeen met het aantal nodige inputs of targets")

    def __str__(self):
        """Deze functie returnt de belangrijke informatie van het neuron netwerk"""
        return "Dit netwerk heeft %s layers, de error van dit netwerk is %s en dit resultaat is behaald na %s epochs" % (len(self.layers), self.loss, self.epoch)