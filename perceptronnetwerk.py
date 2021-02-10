class PerceptronNetwerk:
    def __init__(self, layers: list):
        self.layers = layers
        self.index = 0

    def feed_forward(self, input: [float]):
        """Deze fuctie krijgt een input en haalt het door de layers heen.
            Elke layer veranderd de input naar de output van de vorige layer"""
        volgendeInput = input
        for layer in self.layers:
            volgendeInput = layer.outputFunctie(volgendeInput)
        return volgendeInput