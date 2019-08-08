# utility methods
import tensorflow as tf


def batch_mlp(input, inner_layer_dims, output_dim, variable_scope):
    """Apply MLP to the final axis of a 3D tensor (reusing already defined MLPs).

    Args:
      input: input tensor of shape [B,n,d_in].
      output_sizes: An iterable containing the output sizes of the MLP as defined
          in `basic.Linear`.
      variable_scope: String giving the name of the variable scope. If this is set
          to be the same as a previously defined MLP, then the weights are reused.

    Returns:
      tensor of shape [B,n,d_out] where d_out=output_sizes[-1]
    """
    # Get the shapes of the input and reshape to parallelise across observations

    output = input

    # Pass through MLP
    with tf.variable_scope(variable_scope, reuse=tf.AUTO_REUSE):
        for i, size in enumerate(inner_layer_dims):
            output = tf.nn.relu(
                tf.layers.dense(output, size, name="layer_{}".format(i)))

        # Last layer without a ReLu
        output = tf.layers.dense(output, output_dim, name="layer_{}".format(i + 1))

    return output
