import sys
import numpy as np
import theano
import theano.tensor as T
import network
import prior


class PBP:
    def __init__(self, layer_sizes, mean_y_train, std_y_train):
        """
            Constructor for the class implementing a Bayesian neural network.

            @param layer_sizes  Architecture of the network.
            @param mean_y_train Mean of the training labels
            @param  std_y_train Standard deviation of the training labels.
        """
        var_targets = 1
        self.std_y_train = std_y_train
        self.mean_y_train = mean_y_train

        # Initialize the prior:
        self.prior = prior.Prior(layer_sizes, var_targets)

        # Create the network:
        params = self.prior.get_initial_params()
        self.network = network.Network(params['m_w'], params['v_w'],
                                        params['a'], params['b'])

        # Create the input and output variables in theano:
        self.x = T.vector('x')
        self.y = T.scalar('y')
        
        # Computing the value of logZ, logZ1 and logZ2:
        # (cf: article)
        self.logZ, self.logZ1, self.logZ2 = \
                    self.network.logZ_Z1_Z2(self.x, self.y)

        # Theano function for updating the posterior:
        self.adf_update = theano.function([self.x, self.y], self.logZ,
            updates = self.network.generate_updates(self.logZ, self.logZ1,
            self.logZ2))

        # Theano function for the network probabilistic predictive distribution:
        self.predict_probabilistic = theano.function([self.x],
            self.network.output_probabilistic(self.x))

        # Theano function for the network deterministic predictive distribution:
        self.predict_deterministic = theano.function([ self.x ],
            self.network.output_deterministic(self.x))

    def do_pbp(self, x_train, y_train, n_iterations):
        """
            Function for performing the probabilistic back-propagation.

            @param x_train   The matrix of features for the training data.
            @param y_train   The matrix of labels for the training data.
            @param n_iterations   Number of iterations.
        """
        if n_iterations > 0:
            # First do a single pass:
            self.do_first_pass(x_train, y_train)

            # Refine the prior:
            params = self.network.get_params()
            params = self.prior.refine_prior(params)
            self.network.set_params(params)

            sys.stdout.write('{}\n'.format(0))
            sys.stdout.flush()

            for i in range(int(n_iterations) - 1):
                # Iterates on the passes:
                self.do_first_pass(x_train, y_train)

                # Refine the prior:
                params = self.network.get_params()
                params = self.prior.refine_prior(params)
                self.network.set_params(params)

                sys.stdout.write('{}\n'.format(i + 1))
                sys.stdout.flush()

    def get_deterministic_output(self, x_test):
        """
            Function for making predictions with the Bayesian neural network.

            @param x_test   The matrix of features for the test data.

            @return output       The predicted label for the test target variables.
        """
        output = np.zeros(x_test.shape[0])
        for i in range(x_test.shape[0]):
            output[i] = self.predict_deterministic(x_test[i, :])
            output[i] = output[i] * self.std_y_train + self.mean_y_train

        return output

    def get_predictive_mean_and_variance(self, x_test):
        """
            Function computing the mean and variance of
            the predictions made with the Bayesian neural network.

            @param x_test   The matrix of features for the test data.


            @return mean    The predictive mean for the test target variables.
            @return variance The predictive variance for the test target
                            variables.
            @return v_noise The estimated variance for the additive noise.
        """
        mean = np.zeros(x_test.shape[0])
        variance = np.zeros(x_test.shape[0])

        for i in range(x_test.shape[0]):
            m, v = self.predict_probabilistic(x_test[i, :])
            m = m * self.std_y_train + self.mean_y_train
            v *= self.std_y_train**2
            mean[i] = m
            variance[i] = v

        v_noise = self.network.b.get_value() / \
            (self.network.a.get_value() - 1) * self.std_y_train**2

        return mean, variance, v_noise

    def do_first_pass(self, x, y):
        permutation = np.random.choice(range(x.shape[0]), x.shape[0],
                                        replace=False)

        counter = 0
        for i in permutation:

            old_params = self.network.get_params()
            logZ = self.adf_update(x[i, :], y[i])
            new_params = self.network.get_params()
            self.network.remove_invalid_updates(new_params, old_params)
            self.network.set_params(new_params)

            if counter % 1000 == 0:
                sys.stdout.write('.')
                sys.stdout.flush()

            counter += 1

        sys.stdout.write('\n')
        sys.stdout.flush()

    def sample_w(self):
        self.network.sample_w()
