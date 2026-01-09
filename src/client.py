class Client:
    def __init__(self, client_id, data):
        self.client_id = client_id
        self.x, self.y = data

    def train(self, model, epochs=1, batch_size=32):
        model.fit(self.x, self.y, epochs=epochs, batch_size=batch_size, verbose=0)
        return model.get_weights()
    