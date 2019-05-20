import tensorflow as  tf
import tensorflow_hub as hub
from neural_process import split_context_target, NeuralProcessParams
from neural_process.network import *
from neural_process.loss import *
from neural_process.predict import *
from neural_process.process import *
from neural_process.bert_utils import *
from tensorflow_probability import distributions as tfd

def encoder_h(context_xys: tf.Tensor, params: NeuralProcessParams) -> tf.Tensor:
    """Map context inputs (x_i, y_i) to r_i

    Creates a fully connected network with a single sigmoid hidden layer and linear output layer.

    Parameters
    ----------
    context_xys
        Input tensor, shape: (n_samples, dim_x + dim_y)
    params
        Neural process parameters

    Returns
    -------
        Output tensor of encoder network
    """
    hidden_layer = context_xys
    #     print('hidden layer')
    #     print(hidden_layer)
    #     print(enumerate(params.n_hidden_units_h))
    # First layers are relu
    for i, n_hidden_units in enumerate(params.n_hidden_units_h):
        #         print(i)
        #         print(n_hidden_units)
        hidden_layer = tf.layers.dense(hidden_layer, n_hidden_units,
                                       activation=tf.nn.relu,
                                       name='encoder_layer_{}'.format(i),
                                       reuse=tf.AUTO_REUSE,
                                       kernel_initializer='normal')
    #         print(hidden_layer)

    # Last layer is simple linear
    i = len(params.n_hidden_units_h)
    r = tf.layers.dense(hidden_layer, params.dim_r,
                        name='encoder_layer_{}'.format(i),
                        reuse=tf.AUTO_REUSE,
                        kernel_initializer='normal')
    return r


def aggregate_r(context_rs: tf.Tensor) -> tf.Tensor:
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


def get_z_params(context_r: tf.Tensor, params: NeuralProcessParams) -> GaussianParams:
    """Map encoding to mean and covariance of the random variable Z

    Creates a linear dense layer to map encoding to mu_z, and another linear mapping + a softplus activation for Sigma_z

    Parameters
    ----------
    context_r
        Input encoding tensor, shape: (1, dim_r)
    params
        Neural process parameters

    Returns
    -------
        Output tensors of the mappings for mu_z and Sigma_z
    """
    mu = tf.layers.dense(context_r, params.dim_z, name="z_params_mu", reuse=tf.AUTO_REUSE, kernel_initializer='normal')

    sigma = tf.layers.dense(context_r, params.dim_z, name="z_params_sigma", reuse=tf.AUTO_REUSE,
                            kernel_initializer='normal')
    sigma = tf.nn.softplus(sigma)

    return GaussianParams(mu, sigma)


def decoder_g(input_xs_embedding, z_samples: tf.Tensor, num_labels: int, params: NeuralProcessParams) -> GaussianParams:
    """Determine output y* by decoding input and latent variable

    Creates a fully connected network with a single sigmoid hidden layer and linear output layer.

    Parameters
    ----------
    z_samples
        Random samples from the latent variable distribution, shape: (n_z_draws, dim_z)
    input_xs
        Input values to predict for, shape: (n_x_samples, dim_x)
    params
        Neural process parameters
    noise_std
        Constant standard deviation used on output

    Returns
    -------
        Output tensors for the parameters of Gaussian distributions for target outputy, where its mean mu has shape
        (n_x_samples, n_z_draws)

    """
    # inputs dimensions
    # z_sample has dim [n_draws, dim_z]
    # x_star has dim [N_star, dim_x]

    n_draws = z_samples.get_shape().as_list()[0]
    #     print('n_draws')
    #     print(n_draws)
    n_xs = tf.shape(input_xs_embedding)[0]

    # Repeat z samples for each x*
    z_samples_repeat = tf.expand_dims(z_samples, axis=1)
    z_samples_repeat = tf.tile(z_samples_repeat, [1, n_xs, 1])

    # Repeat x* for each z sample
    #     input_xs_embedding = embedder(input_xs)
    x_star_repeat = tf.expand_dims(input_xs_embedding, axis=0)
    x_star_repeat = tf.tile(x_star_repeat, [n_draws, 1, 1])

    # Concatenate x* and z
    # shape: (n_z_draws, n_xs, dim_x + dim_z)
    inputs = tf.concat([x_star_repeat, z_samples_repeat], axis=2)

    hidden_layer = inputs
    # First layers are relu
    for i, n_hidden_units in enumerate(params.n_hidden_units_g):
        hidden_layer = tf.layers.dense(hidden_layer, n_hidden_units,
                                       activation=tf.nn.relu,
                                       name='decoder_layer_{}'.format(i),
                                       reuse=tf.AUTO_REUSE,
                                       kernel_initializer='normal')

    # Last layer is simple linear
    i = len(params.n_hidden_units_g)
    hidden_layer = tf.layers.dense(hidden_layer, 2 * num_labels,
                                   name='decoder_layer_{}'.format(i),
                                   reuse=tf.AUTO_REUSE,
                                   kernel_initializer='normal')

    # mu will be of the shape [N_star, n_draws]
    # Get the mean an the variance
    mu, log_sigma = tf.split(hidden_layer, 2, axis=-1)

    # Bound the variance
    sigma_star = 0.1 + 0.9 * tf.nn.softplus(log_sigma)
    mu_star = tf.math.sigmoid(mu)

    return GaussianParams(mu_star, sigma_star)


def xy_to_z_params(context_xs: tf.Tensor, context_ys: tf.Tensor,
                   params: NeuralProcessParams) -> GaussianParams:
    """Wrapper to create full network from context samples to parameters of pdf of Z

    Parameters
    ----------
    context_xs
        Tensor with context features, shape: (n_samples, dim_x)
    context_ys
        Tensor with context targets, shape: (n_samples, dim_y)
    params
        Neural process parameters

    Returns
    -------
        Output tensors of the mappings for mu_z and Sigma_z
    """
    #     context_xs = embedder(context_xs)
    xys = tf.concat([context_xs, context_ys], axis=1)
    rs = encoder_h(xys, params)
    r = aggregate_r(rs)
    z_params = get_z_params(r, params)
    return z_params


def loglikelihood(y_star: tf.Tensor, y_pred_params: GaussianParams):
    """Log-likelihood of an output given a predicted """
    p_normal = tfd.MultivariateNormalDiag(loc=y_pred_params.mu,
                                          scale_diag=y_pred_params.sigma)
    loglike = p_normal.log_prob(y_star)
    loglike = tf.reduce_sum(loglike, axis=0)
    loglike = tf.reduce_mean(loglike)
    return loglike


def create_model(input_ids, input_mask, segment_ids, num_labels, scores, params, num_draws, BERT_model_hub):
    #     """Creates a classification model."""

    bert_module = hub.Module(BERT_model_hub, trainable=True)

    bert_inputs = dict(input_ids=input_ids, input_mask=input_mask,
                       segment_ids=segment_ids)

    bert_outputs = bert_module(inputs=bert_inputs, signature="tokens",
                               as_dict=True)

    # Use "pooled_output" for classification tasks on an entire sentence. Use "sequence_outputs" for token-level output.
    output_bert_layer = bert_outputs["pooled_output"]

    tf.logging.info(output_bert_layer)


    # -------
    # #TODO bring this out and set as input
    # params = NeuralProcessParams(dim_r=20, dim_z=20, n_hidden_units_h=[128, 128, 128], n_hidden_units_g=[128, 128, 128])

    btch_sz = tf.shape(output_bert_layer)[0]

    n_context = tf.random_shuffle(tf.range(1, btch_sz))[0]

    indices = tf.range(0, btch_sz)
    context_set_indices = tf.gather(tf.random_shuffle(indices), tf.range(n_context))
    target_set_indices = tf.gather(tf.random_shuffle(indices), tf.range(n_context, btch_sz))

    context_xs = tf.gather(output_bert_layer, context_set_indices)
    context_ys = tf.gather(scores, context_set_indices)
    target_xs = tf.gather(output_bert_layer, target_set_indices)
    target_ys = tf.gather(scores, target_set_indices)

    x_all = output_bert_layer
    y_all = scores

    z_context = xy_to_z_params(context_xs, context_ys, params)
    z_all = xy_to_z_params(x_all, y_all, params)

    epsilon = tf.random_normal([num_draws, params.dim_z])
    z_samples = tf.multiply(epsilon, z_all.sigma)
    z_samples = tf.add(z_samples, z_all.mu)

    y_loss_params = decoder_g(target_xs, z_samples, num_labels,params)

    prior_predict = decoder_g(x_all, epsilon, num_labels, params)
    posterior_predict = decoder_g(x_all, z_samples, num_labels, params)

    loglike = loglikelihood(target_ys, y_loss_params)
    KL_loss = KLqp_gaussian(z_all.mu, z_all.sigma, z_context.mu, z_context.sigma)
    loss = tf.negative(loglike) + KL_loss

    return (loss, posterior_predict, prior_predict, y_all)
