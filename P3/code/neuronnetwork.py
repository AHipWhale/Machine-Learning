class NeuronNetwerk:
    def __init__(self, layers: list):
        self.layers = layers
        self.output = []

    def feed_forward(self, input: [float]):
        """Deze fuctie krijgt een input en haalt het door de layers heen.
            Elke layer veranderd de input naar de output van de vorige layer"""
        volgendeInput = input
        for layer in self.layers:
            volgendeInput = layer.outputFunctie(volgendeInput)
        self.output = volgendeInput
        return volgendeInput

    def __str__(self):
        """Deze functie returnt de belangrijke informatie van het neuron netwerk"""
        return "Dit netwerk heeft %s layers en de output van dit netwerk is %s" % (self.layers, self.output)