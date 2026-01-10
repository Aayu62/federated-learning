import numpy as np

class Server:
    def __init__(self, model):
        self.global_model = model

    def aggregate(self, client_updates):
        weights = [update["weights"] for update in client_updates]
        sizes = [update["size"] for update in client_updates]

        total_size = sum(sizes)
        new_weights = []

        for layer_weights in zip(*weights):
            weighted_sum = np.sum([layer * (size / total_size)
                                   for layer, size in zip(layer_weights, sizes)],
                                  axis=0)
            new_weights.append(weighted_sum)

        self.global_model.set_weights(new_weights)
