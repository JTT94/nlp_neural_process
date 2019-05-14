import tensorflow as tf
import bert

def model_fn_builder(create_model, num_labels, learning_rate, num_train_steps,
                     num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]    
        scores = features["scores"]

        (loss, y, target_ys) = create_model(input_ids, input_mask, segment_ids, num_labels, scores)

        train_op = bert.optimization.create_optimizer(loss, learning_rate, num_train_steps, 
            num_warmup_steps, use_tpu=False)

    # Calculate evaluation metrics.
        def metric_fn(pred_scores, real_scores, trait_num):
            auc_value = tf.metrics.auc(pred_scores[:,trait_num], real_scores[:,trait_num])
            return {"auc"+str(trait_num): auc_value}
        ystar, variance = tf.nn.moments(y.mu,[0])

        labels = tf.math.round(target_ys)
        eval_metrics_lst = [metric_fn(ystar, labels, trait_num) for trait_num in range(num_labels)]
        eval_metrics = {}
        for d in eval_metrics_lst:
            tf.summary.scalar(list(d.keys())[0], list(d.values())[0][1]) #make available to tensorboard
            eval_metrics.update(d)

        if mode == tf.estimator.ModeKeys.TRAIN:
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
        elif mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(predictions={'prediction_mean': ystar, 'prediction_var': variance})
        else:
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metrics)
        # Return the actual model function in the closure
    return model_fn