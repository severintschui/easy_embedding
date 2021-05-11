from embeddings_base import BaseEmbedding

import pennylane as qml
from pennylane import numpy as np


class QAOAEmbedding(BaseEmbedding):
    def __init__(self, feature_dim, n_layers, n_qubits, n_latent_qubits=0, angle_scaling=False, wires=None):
        super().__init__(feature_dim=feature_dim, n_layers=n_layers, n_qubits=n_qubits,
                         n_latent_qubits=n_latent_qubits, wires=wires)

        # number of weights for rotation gates
        self.n_rotation_weights = self.n_total_qubits

        # number of weights for entangling gates
        if self.n_total_qubits == 1:
            self.n_entangling_weights = 0
        elif self.n_total_qubits == 2:
            self.n_entangling_weights = 1
        elif self.n_total_qubits >= 3:
            self.n_entangling_weights = self.n_total_qubits

        # number of weights for scaling the input data
        self.angle_scaling = angle_scaling
        self.n_scaling_weights = int(self.angle_scaling) * self.n_qubits  # 0 or n_qubits

        # setting the total weights per layer
        self.n_weights_per_layer = self.n_rotation_weights + self.n_entangling_weights + self.n_scaling_weights

    def layer(self, features, weights):
        upload = features
        if self.angle_scaling:
            upload = weights[0:self.n_scaling_weights] * upload

        qml.broadcast(unitary=qml.RX, wires=self.feature_wires, pattern='single', parameters=upload)
        qml.broadcast(unitary=qml.Hadamard, wires=self.latent_wires, pattern='single')

        if self.n_total_qubits >= 2:
            qml.broadcast(unitary=qml.MultiRZ, wires=self.wires, pattern='ring',
                          parameters=weights[self.n_scaling_weights:self.n_scaling_weights + self.n_entangling_weights])

        qml.broadcast(unitary=qml.RY, wires=self.wires, pattern='single',
                      parameters=weights[self.n_scaling_weights + self.n_entangling_weights:])

    def layer_starting_weights(self):
        random_weights = np.append(
            np.random.normal(loc=0, scale=1, size=self.n_scaling_weights),
            np.random.uniform(high=4*np.pi, size=self.n_rotation_weights + self.n_entangling_weights)
        )
        return random_weights


class RotationEmbedding(BaseEmbedding):
    def __init__(self, feature_dim, n_layers, n_qubits):
        super().__init__(feature_dim, n_layers, n_qubits, n_features_per_qubit=3)
        self.n_weights_per_layer = 6*self.n_qubits

    def layer(self, features, weights):
        for k in range(self.n_qubits):
            upload = weights[k*6:k*6+3] + weights[k*6+3:(k+1)*6] * features[k*3:(k+1)*3]
            qml.Rot(*upload, wires=self.wires[k])
        qml.broadcast(unitary=qml.CNOT, wires=self.wires, pattern='ring')

    def layer_starting_weights(self):
        random_weights = np.zeros(self.n_weights_per_layer)
        for i in range(self.n_qubits):
            random_weights[i*6+3:(i+1)*6] = np.random.normal(loc=0, scale=1, size=3)
        return random_weights


class CustomEmbedding(BaseEmbedding):
    def __init__(self, feature_dim, n_layers, n_qubits):
        super().__init__(feature_dim, n_layers, n_qubits)

        self.n_weights_per_layer = 2*self.n_qubits

    def layer(self, features, weights):
        for k in range(self.n_qubits):
            qml.RY(weights[k] * features[k], wires=self.wires[k])

        qml.broadcast(qml.CNOT, wires=self.wires, pattern='ring')

        for k in range(self.n_qubits):
            qml.RX(weights[self.n_qubits + k], wires=self.wires[k])

    def layer_starting_weights(self):
        random_weights = np.append(
            np.random.normal(loc=0, scale=1, size=self.n_qubits),
            np.random.uniform(low=0, high=4 * np.pi, size=self.n_qubits)
        )
        return random_weights


if __name__ == '__main__':
    feature_dim = 4
    n_layers = 3
    n_qubits = 1

    embedding = RotationEmbedding(feature_dim, n_layers, n_qubits)
    embedding.draw()
    print(f'Features will be uploaded {embedding.n_uploads} times.')
