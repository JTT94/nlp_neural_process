
import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_hub as hub
import numpy as np
from datetime import datetime
from .params import NeuralProcessParams
from .encoder import Encoder
from .decoder import Decoder
from .bert_utils import create_tokenizer_from_hub_module, create_examples, input_fn_builder, convert_examples_to_features
import bert
from bert import optimization

# This is a path to an uncased (all lowercase) version of BERT
BERT_model_hub = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
tokenizer = create_tokenizer_from_hub_module(BERT_model_hub)


class NLP_NeuralProcess(object):

    def __init__(self,
                 score_col,
                 params=NeuralProcessParams(dim_z=20,
                                            n_hidden_units_h=[128, 128, 128],
                                            n_hidden_units_g=[128, 128, 128]),
                 num_classes=6,
                 num_draws=2,
                 lr=2e-5,
                 batch_size=32,
                 num_warmup_steps=100,
                 num_train_steps=10 ** 3,
                 save_summary_steps=100,
                 save_checkpoints_steps=500,
                 max_seq_length = 128,
                 keep_checkpoint_max=None,
                 output_dir="./test_output",
                 context_features=None
                 ):

        self.params = params
        self.encoder = Encoder(layer_dims=self.params.n_hidden_units_h,
                               latent_dim=self.params.dim_z)
        self.decoder = Decoder(layer_dims=self.params.n_hidden_units_g,
                               num_classes=num_classes)
        self.num_draws = num_draws
        self.num_classes = num_classes
        self.max_seq_length = max_seq_length
        #     self.estimator = None
        # self.embedder = Embedder()
        #####
        num_labels = len(score_col)

        # Specify outpit directory and number of checkpoint steps to save

        run_config = tf.estimator.RunConfig(model_dir=output_dir,
                                            save_summary_steps=save_summary_steps,
                                            save_checkpoints_steps=save_checkpoints_steps)

        model_fn = self.model_fn_builder(num_labels=num_labels, learning_rate=lr,
                                         num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps)

        if context_features is not None:
            estimator_params = {"batch_size": batch_size, "context_features": context_features}
            self.estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config,
                                                    params=estimator_params)

        else:
            self.estimator = tf.estimator.Estimator(model_fn=model_fn, config=run_config,
                                                    params={"batch_size": batch_size})

    def create_model(self,
                     target_input_ids,
                     target_input_mask,
                     target_segment_ids,
                     target_scores=None,
                     context_input_ids=None,
                     context_input_mask=None,
                     context_segment_ids=None,
                     context_scores=None
                     ):

        # apply embedder
        embedder = hub.Module(BERT_model_hub, trainable=False)

        valid_context = (context_input_ids is not None) & (context_input_mask is not None) & (
                    context_segment_ids is not None) & (context_scores is not None)

        # target processing - all scenarios
        target_inputs = dict(input_ids=target_input_ids,
                             input_mask=target_input_mask,
                             segment_ids=target_segment_ids)
        target_embeddings = embedder(inputs=target_inputs,
                                     signature="tokens",
                                     as_dict=True)
        target_xs = target_embeddings["pooled_output"]

        if valid_context:

            # context processing - training
            context_inputs = dict(input_ids=context_input_ids,
                                  input_mask=context_input_mask,
                                  segment_ids=context_segment_ids)
            context_embeddings = embedder(inputs=context_inputs,
                                          signature="tokens",
                                          as_dict=True)
            context_xs = context_embeddings["pooled_output"]
            context_ys = context_scores
            # total x,y
            x_all = tf.concat([context_xs, target_xs], axis=0)

            # get encoding params with context
            context_z_dist = self.encoder(context_xs, context_ys)
            # predictions with context
            posterior_pred = self.decoder(target_xs, context_z_dist.sample(self.num_draws))

            # target scores - context training / evaluation
            if target_scores is not None:
                target_ys = target_scores
                y_all = tf.concat([context_ys, target_ys], axis=0)
                all_z_dist = self.encoder(x_all, y_all)

                # loss
                loglike = self.loglikelihood(target_ys, posterior_pred)
                KL_loss = self.KLqp_gaussian(all_z_dist.parameters['loc'],
                                             all_z_dist.parameters['scale_diag'],
                                             context_z_dist.parameters['loc'],
                                             context_z_dist.parameters['scale_diag'])
                loss = tf.negative(loglike) + KL_loss
                # context and training / evaluation
                return (loss, posterior_pred, target_ys)

            # context prediction
            return (None, posterior_pred, None)

        # no context
        else:
            x_all = target_xs
            # get internal representation
            mean_zero = tf.constant(np.repeat(0., self.params.dim_z))
            epsilon_dist = tfd.MultivariateNormalDiag(loc=mean_zero)
            epsilon = tf.expand_dims(epsilon_dist.sample(self.num_draws), axis=1)
            epsilon = tf.cast(epsilon, tf.float32)
            prior_predict = self.decoder(x_all, epsilon)

            # target scores - no context training / evaluation
            if target_scores is not None:
                target_ys = target_scores
                loglike = self.loglikelihood(target_ys, prior_predict)
                loss = tf.negative(loglike)
                # no context/ training / evaluation
                return (loss, prior_predict, target_ys)

            # no context prediction
            return (None, prior_predict, None)

    def loglikelihood(self, y_star: tf.Tensor, dist):
        """Log-likelihood of an output given a predicted """
        p_normal = tfd.MultivariateNormalDiag(loc=dist.mu, scale_diag=dist.sigma)
        loglike = p_normal.log_prob(y_star)
        loglike = tf.reduce_sum(loglike, axis=0)
        loglike = tf.reduce_mean(loglike)
        return loglike

    def KLqp_gaussian(self, mu_q: tf.Tensor, sigma_q: tf.Tensor, mu_p: tf.Tensor, sigma_p: tf.Tensor) -> tf.Tensor:
        """Kullback-Leibler divergence between two Gaussian distributions

        Determines KL(q || p) = < log( q / p ) >_q

        Parameters
        ----------
        mu_q
            Mean tensor of distribution q, shape: (1, dim)
        sigma_q
            Variance tensor of distribution q, shape: (1, dim)
        mu_p
            Mean tensor of distribution p, shape: (1, dim)
        sigma_p
            Variance tensor of distribution p, shape: (1, dim)

        Returns
        -------
            KL tensor, shape: (1)
        """
        sigma2_q = tf.square(sigma_q) + 1e-16
        sigma2_p = tf.square(sigma_p) + 1e-16
        temp = sigma2_q / sigma2_p + tf.square(mu_q - mu_p) / sigma2_p - 1.0 + tf.log(sigma2_p / sigma2_q + 1e-16)
        return 0.5 * tf.reduce_sum(temp)

    def context_target_split(self, batch_size=32):
        btch_sz = batch_size
        n_context = tf.random_shuffle(tf.range(1, btch_sz))[0]

        indices = tf.range(0, btch_sz)
        context_set_indices = tf.gather(tf.random_shuffle(indices), tf.range(n_context))
        target_set_indices = tf.gather(tf.random_shuffle(indices), tf.range(n_context, btch_sz))

        return context_set_indices, target_set_indices

    def model_fn_builder(self, num_labels, learning_rate, num_train_steps, num_warmup_steps):
        """Returns `model_fn` closure for TPUEstimator."""

        def model_fn(features, mode, params):  # pylint: disable=unused-argument
            """The `model_fn` for TPUEstimator."""

            # run model
            # -------------------------------------------------------------------------------------------
            target_input_ids = None
            target_input_mask = None
            target_segment_ids = None
            target_scores = None

            context_input_ids = None
            context_input_mask = None
            context_segment_ids = None
            context_scores = None

            # training
            if mode == tf.estimator.ModeKeys.TRAIN:
                input_ids = features["input_ids"]
                input_mask = features["input_mask"]
                segment_ids = features["segment_ids"]
                scores = features["scores"]

                # context split
                context_set_indices, target_set_indices = self.context_target_split(batch_size=32)

                target_input_ids = tf.gather(input_ids, target_set_indices)
                target_input_mask = tf.gather(input_mask, target_set_indices)
                target_segment_ids = tf.gather(segment_ids, target_set_indices)
                target_scores = tf.gather(scores, target_set_indices)

                context_input_ids = tf.gather(input_ids, context_set_indices)
                context_input_mask = tf.gather(input_mask, context_set_indices)
                context_segment_ids = tf.gather(segment_ids, context_set_indices)
                context_scores = tf.gather(scores, context_set_indices)


            elif mode == tf.estimator.ModeKeys.PREDICT:
                print('Prediction')
                target_input_ids = features["input_ids"]
                target_input_mask = features["input_mask"]
                target_segment_ids = features["segment_ids"]
                target_scores = features["scores"]

                try:
                    context_input_ids = features["supplied_context_input_ids"][0]
                    context_input_mask = features["supplied_context_input_mask"][0]
                    context_segment_ids = features["supplied_context_segment_ids"][0]
                    context_scores = features["supplied_context_scores"][0]

                except:
                    print("****No context supplied ****")
                    context_input_ids = None
                    context_input_mask = None
                    context_segment_ids = None
                    context_scores = None

            else:
                print('Evaluation')
                target_input_ids = features["input_ids"]
                target_input_mask = features["input_mask"]
                target_segment_ids = features["segment_ids"]
                target_scores = features["scores"]

                try:
                    context_input_ids = features["supplied_context_input_ids"][0]
                    context_input_mask = features["supplied_context_input_mask"][0]
                    context_segment_ids = features["supplied_context_segment_ids"][0]
                    context_scores = features["supplied_context_scores"][0]

                except:
                    print("****No context supplied ****")
                    context_input_ids = None
                    context_input_mask = None
                    context_segment_ids = None
                    context_scores = None

            (loss, prediction, true_y) = self.create_model(target_input_ids,
                                                           target_input_mask,
                                                           target_segment_ids,
                                                           target_scores,
                                                           context_input_ids,
                                                           context_input_mask,
                                                           context_segment_ids,
                                                           context_scores)

            train_op = optimization.create_optimizer(loss, learning_rate, num_train_steps, num_warmup_steps,
                                                          use_tpu=False)
            ystar, _ = tf.nn.moments(prediction.mu, [0])
            variance, _ = tf.nn.moments(tf.math.square(prediction.sigma), [0])

            # Calculate evaluation metrics
            eval_metrics = {}

            # AUC
            def metric_fn(pred_scores, real_scores, trait_num):
                auc_value = tf.metrics.auc(real_scores[:, trait_num], pred_scores[:, trait_num])
                accuracy_value = tf.metrics.accuracy(labels=tf.round(real_scores[:, trait_num]),
                                                     predictions=tf.round(pred_scores[:, trait_num]))
                recall_value = tf.metrics.recall(labels=tf.round(real_scores[:, trait_num]),
                                                 predictions=tf.round(pred_scores[:, trait_num]))
                precision_value = tf.metrics.precision(labels=tf.round(real_scores[:, trait_num]),
                                                       predictions=tf.round(pred_scores[:, trait_num]))
                return {"auc" + str(trait_num): auc_value, "accuracy" + str(trait_num): accuracy_value,
                        "recall" + str(trait_num): recall_value, "precision" + str(trait_num): precision_value}

            labels = true_y  # need to round them if true labels are not 1 or 0
            eval_metrics_lst = [metric_fn(ystar, labels, trait_num) for trait_num in range(num_labels)]

            for d in eval_metrics_lst:
                tf.summary.scalar(list(d.keys())[0], list(d.values())[0][1])  # make available to tensorboard
                tf.summary.scalar(list(d.keys())[1], list(d.values())[1][1])  # make available to tensorboard
                eval_metrics.update(d)

            # output from model
            # -------------------------------------------------------------------------------------------

            # training
            if mode == tf.estimator.ModeKeys.TRAIN:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

            # prediction
            elif mode == tf.estimator.ModeKeys.PREDICT:
                return tf.estimator.EstimatorSpec(mode=mode,
                                                  predictions={'prediction_mean': ystar, 'prediction_var': variance})

            # evaluation
            else:
                return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics)

        return model_fn

    def prepare_examples(self, df, score_column, text_col_name, supplied_context_df=None):
        num_labels = len(score_column)
        input_examples = create_examples(df, score_column, text_col_name)
        input_features = convert_examples_to_features(input_examples, self.max_seq_length, tokenizer)

        if supplied_context_df is not None:
            supplied_context_examples = create_examples(supplied_context_df, score_column, text_col_name)
            supplied_context_features = convert_examples_to_features(supplied_context_examples, self.max_seq_length,
                                                                     tokenizer)
            input_fn = input_fn_builder(
                features=input_features, seq_length=self.max_seq_length,
                num_labels=num_labels, is_training=True, drop_remainder=False,
                supplied_context_features=supplied_context_features)

        else:
            input_fn = input_fn_builder(
                features=input_features, seq_length=self.max_seq_length,
                num_labels=num_labels, is_training=True, drop_remainder=False)

        return input_fn

    def predict(self,
                df,
                score_col,
                text_col,
                supplied_context_df=None
                ):

        pred_input_fn = self.prepare_examples(df, score_col, text_col, supplied_context_df)
        preds = self.estimator.predict(input_fn=pred_input_fn)

        return preds

    def evaluate(self,
                 eval_steps,
                 df,
                 score_col,
                 text_col,
                 supplied_context_df=None):

        eval_input_fn = self.prepare_examples(df, score_col, text_col, supplied_context_df)

        result = self.estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        return result

    def train(self, num_train_steps,
              df_train,
              score_col,
              text_col,
              ):

        # Create an input function for training. drop_remainder = True for using TPUs.
        train_input_fn = self.prepare_examples(df_train, score_col, text_col)

        print('Beginning Training!')
        current_time = datetime.now()
        self.estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
        print("Training took time ", datetime.now() - current_time)

    def train_and_evaluate(self,
                           df_train,
                           df_eval,
                           score_col,
                           text_col,
                           num_train_steps,
                           eval_steps,
                           supplied_context_df=None):

        train_input_fn = self.prepare_examples(df_train, score_col, text_col)
        eval_input_fn = self.prepare_examples(df_eval, score_col, text_col, supplied_context_df)

        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=num_train_steps)
        eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn, steps=eval_steps)

        result = tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)
        return result
