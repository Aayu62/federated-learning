import numpy as np
import tensorflow as tf
from collections import defaultdict

def load_and_split_data(num_clients=5, shards_per_client=2):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Create Non-IID label shards
    label_indices = defaultdict(list)
    for idx, label in enumerate(y_train):
        label_indices[int(label)].append(idx)

    # Each client gets data from only 2 labels
    client_data = []
    labels_per_client = 10 // num_clients

    for client_id in range(num_clients):
        allowed_labels = list(range(
            client_id * labels_per_client,
            (client_id + 1) * labels_per_client
        ))

        indices = []
        for label in allowed_labels:
            indices.extend(label_indices[label][:200]) #200 samples per label

        
        client_x = x_train[indices]
        client_y = y_train[indices]
        client_data.append((client_x, client_y))
    
    return client_data, (x_test, y_test)