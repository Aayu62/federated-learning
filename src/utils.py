import numpy as np
import tensorflow as tf

def load_and_split_data(num_clients=5):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Split training data evenly among clients
    client_data = []
    split_size = len(x_train) // num_clients

    for i in range(num_clients):
        start = i * split_size
        end = start + split_size
        client_x = x_train[start:end]
        client_y = y_train[start:end]
        client_data.append((client_x, client_y))
    
    return client_data, (x_test, y_test)