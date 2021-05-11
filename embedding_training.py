import pennylane as qml
from pennylane import numpy as np
from embeddings_base import BaseEmbedding

class EmbeddingTrainer:
    """Interface for training of a quantum embedding instance.

    Provides a simple high-level interface for training a quantum embedding using a fidelity cost function.
    Additionally, it can generate a prediction function based on weights learned from a training procedure and the
    supplied embedding circuit.

    Attributes:
        X (np.array): Array of shape (n_datapoints, feature_dim) containing floats. Stores the training dataset.
        y (np.array): Array of shape (n_datapoints,). Stores the training labels.
        embedding (BaseEmbedding): Instance of BaseEmbedding defining a custom quantum feature embedding.
        n_datapoints (int): Number of datapoints in the training dataset.
        feature_dim (int): Dimensionality of the training dataset.
        data_classes (np.array): Storing the different types of data classes. E.g. ['class A', 'class B']
        n_data_classes (int): Number of data classes. NOTE: Must be 2, as multi-class problems are not supported yet.
        class_priors (np.array): Contains prior probabilities of the data classes.
        opt (pennylane.optimizer): Contains instance of pennylane optimizer.
    """

    def __init__(self, embedding, X, y):
        """
        Args:
            embedding (BaseEmbedding): Instance of BaseEmbedding
            X (np.array): Training dataset of shape (n_datapoints, feature_dim)
            y (np.array): Training labels of shape (n_datapoints,)
        """
        # check if the embedding object is of the correct type
        if not isinstance(embedding, BaseEmbedding):
            raise ValueError('Embedding must be an instance that inherits from BaseEmbedding class.')

        self.embedding = embedding

        # check if the dataset X is of the correct shape
        if len(np.shape(X)) != 2:
            raise ValueError(f'Dataset X must be array of shape (n_datapoints, feature_dim), was given {np.shape(X)}.')

        self.n_datapoints, self.feature_dim = np.shape(X)

        # check if the dataset X and the training labels y have the same first dimension
        if self.n_datapoints != np.shape(y)[0]:
            raise ValueError(f'Dataset X and training labels y must have the same first dimension. Got {self.n_datapoints} datapoints and {np.shape(y)[0]} labels.')

        # check if the dataset feature dimension matches the embedding feature dimension
        if self.feature_dim != embedding.feature_dim:
            raise ValueError(f'Dataset dimension does not match embedding feature dimension! Expected d={embedding.feature_dim}, was given d={self.feature_dim}.')

        self.X = np.array(X, dtype=float, requires_grad=False)
        self.y = np.array(y, requires_grad=False)

        self.data_classes = np.unique(self.y)
        self.n_data_classes = len(self.data_classes)
        self.class_priors = np.array([len(X[self.y == data_class])/self.n_datapoints for data_class in self.data_classes])

        if self.n_data_classes > 2:
            raise NotImplementedError('EmbeddingTrainer currently only supports 2-class classification thesis_datasets!')

        self.X_1 = self.X[self.y == self.data_classes[0]]
        self.X_2 = self.X[self.y == self.data_classes[1]]

        self.opt = None


    def train(self, n_epochs, batch_size, learning_rate=0.01, starting_weights=None, compute_cost=True):
        """Train the embedding with given hyperparameters.

        Args:
            n_epochs (int): Number of times the optimizer is exposed to the whole training dataset.
            batch_size (int): Size of batches to use when iterating over training dataset.
            learning_rate (float): Learning rate (stepsize) of the optimizer.
            starting_weights (np.array, optional): If supplied the optimizer will start from given weights. Otherwise,
                random weights will be generated using the BaseEmbedding.random_starting_weights method.

        Returns:
            Tuple containing final weights, weights after each epoch and cost after each epoch.
        """
        self.opt = qml.AdamOptimizer(stepsize=learning_rate)
        return self._train(n_epochs, batch_size, starting_weights, compute_cost)

    def _train(self, n_epochs, batch_size, starting_weights, compute_cost):
        if starting_weights is None:
            weights = self.embedding.random_starting_weights()
        else:
            if np.shape(starting_weights) != self.embedding.weight_shape:
                raise ValueError(f'Starting weights must have shape {self.embedding.weight_shape}, was given {np.shape(starting_weights)}.')
            weights = starting_weights

        weights_history = np.zeros((n_epochs+1, ) + self.embedding.weight_shape)
        weights_history[0] = weights

        if compute_cost:
            cost_history = np.zeros(n_epochs+1)
            cost_history[0] = self.cost(weights, self.X_1, self.X_2)

        # expose the embedding 'n_epochs' times to the whole dataset
        for i in range(n_epochs):
            # iterate over batched data
            for input_batch, target_batch in self.__iterate_minibatches(batch_size):
                filter = target_batch == self.data_classes[0]
                inputs_1 = input_batch[filter]
                inputs_2 = input_batch[np.logical_not(filter)]
                weights = self.opt.step(self.cost, weights, inputs_1=inputs_1, inputs_2=inputs_2)

            weights_history[i+1] = weights
            if compute_cost:
                cost_history[i+1] = self.cost(weights, self.X_1, self.X_2)
                print(f'Cost after epoch {i + 1}: {cost_history[i]}')
            else:
                print(f'Completed epoch {i+1}')

        if compute_cost:
            return weights, (weights_history, cost_history)
        else:
            return weights, weights_history

    def cost(self, weights, inputs_1=None, inputs_2=None):
        ensemble_1 = self.embedding.generate_ensemble(inputs_1, weights)
        ensemble_2 = self.embedding.generate_ensemble(inputs_2, weights)
        observable = self.class_priors[0] * ensemble_1 - self.class_priors[1] * ensemble_2
        return 1 - np.real(np.trace(observable @ observable))

    def __iterate_minibatches(self, batch_size, shuffle=True):
        """ Helper function to iterate over batched dataset

        Args:
            targets (np.array): Array of target labels.
            batch_size (int): Size of each batch
            shuffle (bool): If True, will shuffle indices and thus iterate over the training data in a random order.
        """
        indices = np.arange(self.n_datapoints)
        if shuffle:
            np.random.shuffle(indices)
        for start_idx in range(0, self.n_datapoints - batch_size + 1, batch_size):
            excerpt = indices[start_idx:start_idx + batch_size]
            yield self.X[excerpt], self.y[excerpt]