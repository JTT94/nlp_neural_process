import tensorflow as tf
from .neural_network_utils import batch_mlp
from .params import GaussianParams


class Decoder(object):
    """The Decoder."""

    def __init__(self, layer_dims, num_classes):
        self.layer_dims = layer_dims
        self.num_classes = num_classes

    def __call__(self, input_xs_embedding, z_samples):
        # inputs dimensions
        # z_sample has dim [n_draws, dim_z]
        # x_star has dim [N_star, dim_x]
        n_draws = z_samples.get_shape().as_list()[0]
        n_xs = tf.shape(input_xs_embedding)[0]

        # Repeat z samples for each x*
        # z_samples_repeat = tf.expand_dims(z_samples, axis=1)

        # z_samples_repeat = tf.expand_dims(z_samples, axis=1)
        z_samples_repeat = tf.tile(z_samples, [1, n_xs, 1])

        # Repeat x* for each z sample
        x_star_repeat = tf.expand_dims(input_xs_embedding, axis=0)
        x_star_repeat = tf.tile(x_star_repeat, [n_draws, 1, 1])

        # Concatenate x* and z
        inputs = tf.concat([x_star_repeat, z_samples_repeat], axis=2)

        # decoder mlp
        inner_layer_dims = self.layer_dims
        output_dim = self.num_classes * 2
        hidden = batch_mlp(inputs, inner_layer_dims, output_dim, "decoder")

        # Get the mean an the variance
        mu, log_sigma = tf.split(hidden, 2, axis=-1)

        # Bound the variance
        sigma_star = 0.1 + 0.9 * tf.nn.softplus(log_sigma)
        mu_star = tf.math.sigmoid(mu)

        return GaussianParams(mu_star, sigma_star)


