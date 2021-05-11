from abc import ABC, abstractmethod

import pennylane as qml
from pennylane import numpy as np


class AbstractBaseEmbedding(ABC):
    @abstractmethod
    def apply(self, features, weights):
        pass

    @abstractmethod
    def layer(self, features, weights):
        pass

    @abstractmethod
    def layer_starting_weights(self):
        pass

    @abstractmethod
    def random_starting_weights(self):
        pass

    @abstractmethod
    def draw(self):
        pass

    @abstractmethod
    def feature_map(self, X, weights, return_type):
        pass

    @abstractmethod
    def generate_model_function(self, X, weights):
        pass

    @abstractmethod
    def generate_prediction_function(self, X, weights):
        pass

    @abstractmethod
    def generate_ensemble(self, X, weights):
        pass

    @abstractmethod
    def generate_observable(self, X_1, X_2, weights):
        pass

    @abstractmethod
    def _feature_padding(self, features):
        pass

    @abstractmethod
    def _embedding_circuit_model(self, observable):
        pass

    @property
    @abstractmethod
    def _embedding_circuit_vector(self):
        pass

    @property
    @abstractmethod
    def _embedding_circuit_matrix(self):
        pass

    @property
    @abstractmethod
    def _device(self):
        pass

    @property
    @abstractmethod
    def feature_dim(self):
        pass

    @property
    @abstractmethod
    def wires(self):
        pass

    @property
    @abstractmethod
    def feature_wires(self):
        pass

    @property
    @abstractmethod
    def latent_wires(self):
        pass

    @property
    @abstractmethod
    def n_layers(self):
        pass

    @property
    @abstractmethod
    def n_sub_layers(self):
        pass

    @property
    @abstractmethod
    def n_repeats_per_layer(self):
        pass

    @property
    @abstractmethod
    def n_qubits(self):
        pass

    @property
    @abstractmethod
    def n_features_per_qubit(self):
        pass

    @property
    @abstractmethod
    def n_weights_per_layer(self):
        pass

    @property
    @abstractmethod
    def n_latent_qubits(self):
        pass

    @property
    @abstractmethod
    def n_total_qubits(self):
        pass

    @property
    @abstractmethod
    def weight_shape(self):
        pass

    @property
    @abstractmethod
    def feature_shape(self):
        pass

    @property
    @abstractmethod
    def hilbert_space_dim(self):
        pass

    @property
    @abstractmethod
    def padding_width(self):
        pass

    @property
    @abstractmethod
    def n_uploads(self):
        pass


class BaseEmbedding(AbstractBaseEmbedding):
    def __init__(self, feature_dim, n_layers, n_qubits, n_features_per_qubit=1, n_latent_qubits=0, wires=None):
        self.__feature_dim = self.__type_checker(feature_dim, int, 'feature_dim')
        self.__n_layers = self.__type_checker(n_layers, int, 'n_layers')
        self.__n_qubits = self.__type_checker(n_qubits, int, 'n_qubits')
        self.__n_features_per_qubit = self.__type_checker(n_features_per_qubit, int, 'n_qubits')
        self.__n_latent_qubits = self.__type_checker(n_latent_qubits, int, 'n_latent_qubits')

        if wires is None:
            self.__wires = list(range(self.n_total_qubits))
        elif isinstance(wires, int):
            self.__wires = list(range(wires))
        elif isinstance(wires, list):
            self.__wires = wires
        elif isinstance(wires, range):
            self.__wires = list(wires)
        else:
            raise ValueError('Wires argument needs to be None, a list, range-object or an integer')

        if len(self.__wires) != self.n_total_qubits:
            raise ValueError('Number of wires must match number of total qubits!')

        self.__n_sub_layers = int(np.ceil(self.feature_dim / (self.n_qubits * self.n_features_per_qubit)))

        if self.n_sub_layers > 1:
            self.__n_repeats_per_layer = 1
            self.__padding_width = self.n_features_per_qubit * self.n_qubits * self.n_sub_layers - self.feature_dim
        else:
            self.__n_repeats_per_layer = int(np.floor(self.n_features_per_qubit * self.n_qubits / self.feature_dim))
            self.__padding_width = (self.n_features_per_qubit * self.n_qubits) % self.feature_dim

    @qml.template
    def apply(self, features, weights):
        if np.shape(features) != self.feature_shape:
            raise ValueError(f'Feature must have shape {self.feature_shape}, was given {np.shape(features)}')
        if np.shape(weights) != self.weight_shape:
            raise ValueError(f'Weights must have shape {self.weight_shape}, was given {np.shape(weights)}')

        features = self._feature_padding(features)
        fpl = self.n_features_per_qubit * self.n_qubits     # features per layer
        for i in range(self.n_layers):
            for j in range(self.n_sub_layers):
                self.layer(features[j*fpl:(j+1)*fpl], weights[i, j, :])

    def feature_map(self, X, weights, return_type='vector'):
        if return_type == 'vector':
            circ = self._embedding_circuit_vector
        elif return_type == 'matrix':
            circ = self._embedding_circuit_matrix
        else:
            raise ValueError("'return_type' must be either 'vector' or 'matrix'")

        if np.shape(X) == (self.feature_dim,):
            return np.array([circ(X, weights)])
        elif len(np.shape(X)) == 2 and np.shape(X)[1] == self.feature_dim:
            return np.array([circ(x, weights) for x in X])
        else:
            raise ValueError(f'Argument must be either a single feature vector of length {self.feature_dim} or a '
                             f'collection of vectors of shape (*, {self.feature_dim})!')

    def generate_model_function(self, observable, weights):
        circuit = self._embedding_circuit_model(observable)

        def model_function(X):
            if np.shape(X) == (self.feature_dim,):
                return np.array([circuit(X, weights)])
            elif len(np.shape(X)) == 2 and np.shape(X)[1] == self.feature_dim:
                return np.array([circuit(x, weights) for x in X])
            else:
                raise ValueError(f'X must be either a single feature vector of length {self.feature_dim} or a '
                                 f'collection of vectors of shape (*, {self.feature_dim})!')

        return model_function

    def generate_prediction_function(self, observable, weights, target_labels=None):
        if target_labels is None:
            target_labels = [0, 1]

        model_function = self.generate_model_function(observable, weights)

        def prediction_function(X):
            model_output = model_function(X)
            predictions = np.array(model_output < 0, dtype=int)
            return np.array([target_labels[prediction] for prediction in predictions])

        return prediction_function

    def generate_ensemble(self, X, weights):
        output = self.feature_map(X, weights, return_type='matrix')
        if len(output) > 0:
            return np.sum(output, axis=0) / len(output)
        else:
            return np.zeros((self.hilbert_space_dim, self.hilbert_space_dim))

    def generate_observable(self, X_1, X_2, weights):
        prior = len(X_1) / (len(X_1) + len(X_2))
        return prior * self.generate_ensemble(X_1, weights) - (1-prior) * self.generate_ensemble(X_2, weights)

    def random_starting_weights(self):
        random_weights = np.zeros(self.weight_shape, requires_grad=True)
        for i in range(self.weight_shape[0]):
            for j in range(self.weight_shape[1]):
                random_weights[i,j] = self.layer_starting_weights()
        return random_weights

    def draw(self):
        drawer = qml.draw(self._embedding_circuit_vector)
        dummy_features = np.ones(self.feature_shape)
        dummy_weights = np.ones(self.weight_shape)
        print(drawer(dummy_features, dummy_weights))

    def _feature_padding(self, features):
        res = features
        for i in range(self.n_repeats_per_layer-1):
            res = np.append(res, features)
        return np.append(res, np.zeros(self.padding_width))

    @property
    def n_weights_per_layer(self):
        if hasattr(self, '_BaseEmbedding__n_weights_per_layer'):
            return self.__n_weights_per_layer
        else:
            raise NotImplementedError(f'The attribute n_weights_per_layer must be defined')

    @n_weights_per_layer.setter
    def n_weights_per_layer(self, value):
        self.__n_weights_per_layer = self.__type_checker(value, int, 'n_weights_per_layer')

    @property
    def feature_dim(self):
        return self.__feature_dim

    @property
    def wires(self):
        return self.__wires

    @property
    def n_layers(self):
        return self.__n_layers

    @property
    def n_qubits(self):
        return self.__n_qubits

    @property
    def n_latent_qubits(self):
        return self.__n_latent_qubits

    @property
    def n_features_per_qubit(self):
        return self.__n_features_per_qubit

    @property
    def feature_wires(self):
        return self.wires[0:self.n_qubits]

    @property
    def latent_wires(self):
        return self.wires[self.n_qubits:]

    @property
    def n_sub_layers(self):
        return self.__n_sub_layers

    @property
    def n_repeats_per_layer(self):
        return self.__n_repeats_per_layer

    @property
    def n_total_qubits(self):
        return self.__n_qubits + self.__n_latent_qubits

    @property
    def feature_shape(self):
        return (self.feature_dim,)

    @property
    def weight_shape(self):
        if not hasattr(self, '_BaseEmbedding__weight_shape'):
            self.__weight_shape = (self.n_layers, self.n_sub_layers, self.n_weights_per_layer)
        return self.__weight_shape

    @property
    def hilbert_space_dim(self):
        return 2**self.n_total_qubits

    @property
    def padding_width(self):
        return self.__padding_width

    @property
    def n_uploads(self):
        return self.n_repeats_per_layer * self.n_layers

    @property
    def _device(self):
        if not hasattr(self, '_BaseEmbedding__device'):
            self.__device = qml.device('default.qubit', self.wires)
        return self.__device

    def _embedding_circuit_model(self, observable):
        operator = qml.Hermitian(observable, wires=self.wires)
        @qml.qnode(self._device)
        def circuit(features, weights):
            self.apply(features, weights)
            return qml.expval(operator)
        return circuit


    @property
    def _embedding_circuit_vector(self):
        if not hasattr(self, '_BaseEmbedding__embedding_circuit_vector'):
            @qml.qnode(self._device)
            def circuit(features, weights):
                self.apply(features, weights)
                return qml.state()

            self.__embedding_circuit_vector = circuit
        return self.__embedding_circuit_vector

    @property
    def _embedding_circuit_matrix(self):
        if not hasattr(self, '_BaseEmbedding__embedding_circuit_matrix'):
            @qml.qnode(self._device)
            def circuit(features, weights):
                self.apply(features, weights)
                return qml.density_matrix(self.wires)
            self.__embedding_circuit_matrix = circuit
        return self.__embedding_circuit_matrix

    def __type_checker(self, value, dtype, name):
        if isinstance(value, dtype):
            return value
        else:
            raise ValueError(f'{name} must be of type {dtype}, was given {type(value)}')




if __name__ == '__main__':
    # trying to initiate an instance of BaseEmbedding will fail
    feature_dim = 10
    n_layers = 8
    n_qubits = 5

    try:
        base_embedding = BaseEmbedding(feature_dim, n_layers, n_qubits)
    except Exception as ex:
        print(f'Initiating of BaseEmbedding failed!\nError message: {ex}')