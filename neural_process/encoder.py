import tensorflow as tf
from tensorflow_probability import distributions as tfd
from .neural_network_utils import batch_mlp
from .params import GaussianParams


class Encoder(object):

    def __init__(self, layer_dims, latent_dim):
        self.layer_dims = layer_dims
        self.latent_dim = latent_dim

    def __call__(self, xs, ys):
        print(xs)
        print(ys)
        xys = tf.concat([xs, ys], axis=1)

        # encoder mlp
        inner_layer_dims = self.layer_dims[:-1]
        output_dim = self.layer_dims[-1]
        rs = batch_mlp(xys, inner_layer_dims, output_dim, "encoder")

        # aggregate rs
        r = self._aggregate_r(rs)

        # get mu and sigma
        z_params = self._get_z_params(r)

        # distribution
        dist = tfd.MultivariateNormalDiag(loc=z_params.mu,
                                          scale_diag=z_params.sigma)
        return dist

    def _aggregate_r(self, context_rs: tf.Tensor) -> tf.Tensor:
        """Aggregate the output of the encoder to a single representation

        Creates an aggregation (mean) operator to combine the encodings of multiple context inputs

        Parameters
        ----------
        context_rs
            Input encodings tensor, shape: (n_samples, dim_r)

        Returns
        -------
            Output tensor of aggregation result
        """
        mean = tf.reduce_mean(context_rs, axis=0)
        r = tf.reshape(mean, [1, -1])
        return r

    def _get_z_params(self, context_r: tf.Tensor) -> GaussianParams:
        """Map encoding to mean and covariance of the random variable Z

        Creates a linear dense layer to map encoding to mu_z, and another linear mapping +
         a softplus activation for Sigma_z

        Parameters
        ----------
        context_r
            Input encoding tensor, shape: (1, dim_r)

        Returns
        -------
            Output tensors of the mappings for mu_z and Sigma_z
        """
        hidden = context_r
        with tf.variable_scope("latent_encoder", reuse=tf.AUTO_REUSE):
            # First apply intermediate relu layer
            hidden = tf.nn.relu(
                tf.layers.dense(hidden,
                                (self.layer_dims[-1] + self.latent_dim) / 2,
                                name="penultimate_layer"))

            # Then apply further linear layers to output latent mu and log sigma
            mu = tf.layers.dense(hidden, self.latent_dim, name="mean_layer")
            log_sigma = tf.layers.dense(hidden, self.latent_dim, name="std_layer")

        # Compute sigma
        sigma = 0.1 + 0.9 * tf.sigmoid(log_sigma)

        return GaussianParams(mu, sigma)
