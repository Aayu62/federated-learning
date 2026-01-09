from model import create_model
from utils import load_and_split_data
from client import Client
from server import Server

def run_federated_learning(rounds=5, num_clients=5):
    client_data, test_data = load_and_split_data(num_clients)
    x_test, y_test = test_data

    server = Server(create_model())

    clients = [Client(i, client_data[i]) for i in range(num_clients)]

    for r in range(rounds):
        print(f"Round {r+1} started")

        client_weights = []

        for client in clients:
            model_copy = create_model()
            model_copy.set_weights(server.global_model.get_weights())

            updated_weights = client.train(model_copy)
            client_weights.append(updated_weights)

        server.aggregate(client_weights)

        loss, acc = server.global_model.evaluate(x_test, y_test, verbose=0)
        print(f"-> Round {r+1}: test accuracy = {acc:.4f}")

    print("Training completed!")

if __name__ == "__main__":
    run_federated_learning()