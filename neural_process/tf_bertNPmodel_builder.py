import tensorflow as tf
import bert

def model_fn_builder(create_model, num_labels, learning_rate, NPparams, BERT_model_hub, num_train_steps,
                     num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        scores = features["scores"]

        (loss, posterior_predict, prior_predict, true_y) = create_model(input_ids, input_mask, segment_ids, num_labels,
                                                                        scores, NPparams, BERT_model_hub)

        train_op = bert.optimization.create_optimizer(loss, learning_rate, num_train_steps,
                                                      num_warmup_steps, use_tpu=False)

        # output from model
        ystar, variance = tf.nn.moments(posterior_predict.mu, [0])
        prior_ystar, prior_variance = tf.nn.moments(prior_predict.mu, [0])

        # Calculate evaluation metrics.
        eval_metrics = {}

        # AUC
        def metric_fn(pred_scores, real_scores, trait_num):
            auc_value = tf.metrics.auc(real_scores[:, trait_num], pred_scores[:, trait_num])
            accuracy_value = tf.metrics.accuracy(labels=tf.round(real_scores[:, trait_num]),
                                                 predictions=tf.round(pred_scores[:, trait_num]))
            return {"auc" + str(trait_num): auc_value, "accuracy" + str(trait_num): accuracy_value}

        labels = true_y  # need to round them if true labels are not 1 or 0
        eval_metrics_lst = [metric_fn(prior_ystar, labels, trait_num) for trait_num in range(num_labels)]

        for d in eval_metrics_lst:
            tf.summary.scalar(list(d.keys())[0], list(d.values())[0][1])  # make available to tensorboard
            eval_metrics.update(d)

        # returns
        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions={'prediction_mean': prior_ystar,
                                                                      'prediction_var': prior_variance})
        else:
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics)
        # Return the actual model function in the closure

    return model_fn