
import numpy as np
import pickle
import gzip
import pbp

class PBP_net:
    def __init__(self, x_train, y_train, n_hidden, n_epochs=40,
                    normalize=False):
        """
            Constructor for the class implementing a Bayesian neural network
            trained with the probabilistic back propagation method.

            @param x_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_hidden     Vector with the number of neurons for each
                                hidden layer.
            @param n_epochs     Number of epochs for which to train the
                                network. The recommended value 40 should be
                                enough.
            @param normalize    Whether to normalize the input features. This
                                is recommended unless the input vector is for
                                example formed by binary features (a
                                fingerprint). In that case we do not recommend
                                to normalize the features.
        """
        # Normalize the training data if necessary:
        if normalize:
            self.std_x_train = np.std(x_train, 0)
            self.std_x_train[self.std_x_train == 0] = 1
            self.mean_x_train = np.mean(x_train, 0)
        else:
            self.std_x_train = np.ones(x_train.shape[1])
            self.mean_x_train = np.zeros(x_train.shape[1])

        x_train = (x_train - np.full(x_train.shape, self.mean_x_train)) / \
            np.full(x_train.shape, self.std_x_train)

        self.mean_y_train = np.mean(y_train)
        self.std_y_train = np.std(y_train)

        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train

        # Construct the network (ie: create a PBP object)
        # Layer 1: (dim_x_train x n_hidden[1])
        # Hidden layer l: (n_hidden[l-1] x n_hidden[l])
        # Output layer: (n_hidden[L] x 1)
        n_units_per_layer = \
            np.concatenate(([x_train.shape[1] ], n_hidden, [1]))

        self.pbp_instance = \
            pbp.PBP(n_units_per_layer, self.mean_y_train, self.std_y_train)

        # Iterate the learning process:
        self.pbp_instance.do_pbp(x_train, y_train_normalized, n_epochs)

    def re_train(self, x_train, y_train, n_epochs):
        """
            Function that re-trains the network on some data.

            @param x_train      Matrix with the features for the training data.
            @param y_train      Vector with the target variables for the
                                training data.
            @param n_epochs     Number of epochs for which to train the
                                network. 
        """
        # Normalize the training data:
        x_train = (x_train - np.full(x_train.shape, self.mean_x_train)) / \
            np.full(x_train.shape, self.std_x_train)
        y_train_normalized = (y_train - self.mean_y_train) / self.std_y_train

        # Iterate the learning process:
        self.pbp_instance.do_pbp(x_train, y_train_normalized, n_epochs)

    def predict(self, x_test):
        """
            Function for making predictions with the Bayesian neural network.

            @param x_test   The matrix of features for the test data
            
    
            @return m       The predictive mean for the test target variables.
            @return v       The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.
        """
        # Convert format of test set:
        x_test = np.array(x_test, ndmin= 2)

        # Normalize the test set:
        x_test = (x_test - np.full(x_test.shape, self.mean_x_train)) / \
            np.full(x_test.shape, self.std_x_train)

        # Predictive mean and variance for the test data:
        m, v, v_noise = self.pbp_instance.get_predictive_mean_and_variance(x_test)

        return m, v, v_noise

    def predict_deterministic(self, x_test):
        """
            Function for making deterministic predictions with the Bayesian neural network.

            @param x_test   The matrix of features for the test data


            @return o       The predictive value for the test target variables.
        """
        # Convert format of test set:
        x_test = np.array(x_test, ndmin = 2)

        # Normalize the test set:
        x_test = (x_test - np.full(x_test.shape, self.mean_x_train)) / \
            np.full(x_test.shape, self.std_x_train)

        # Compute predictive mean and variance for the test data:
        o = self.pbp_instance.get_deterministic_output(x_test)

        return o

    def sample_weights(self):
        """
            Function that draws a sample from the posterior approximation
            to the weights distribution.
        """
        self.pbp_instance.sample_w()

    def save_to_file(self, filename):
        """
            Function that stores the network in a file.

            @param filename   The name of the file.
        """
        # Pickle save the network:
        def save_object(obj, filename):

            result = pickle.dumps(obj)
            with gzip.GzipFile(filename, 'wb') as dest: dest.write(result)
            dest.close()

        save_object(self, filename)

def load_PBP_net_from_file(filename):
    """
        Function that load a network from a file.

        @param filename   The name of the file.
    """
    def load_object(filename):

        with gzip.GzipFile(filename, 'rb') as \
            source: result = source.read()
        ret = pickle.loads(result)
        source.close()

        return ret

    # Load the dictionary with the network parameters:
    pbp_network = load_object(filename)

    return pbp_network
