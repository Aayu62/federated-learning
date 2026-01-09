import numpy as np

class Server:
    def __init__(self, model):
        self.global_model = model

    def aggregate(self, client_weights):
        new_weights = []
        for weights_list_tuple in zip(*client_weights):
            new_weights.append(
                np.mean(np.array(weights_list_tuple), axis=0)
            )
        self.global_model.set_weights(new_weights)