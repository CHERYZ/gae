import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class OptimizerAE(object):
    def __init__(self, preds, labels, pos_weight, norm):
        preds_sub = preds
        labels_sub = labels

        # cross_entropy 交叉熵
        # logits 代表模型的输出.
        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class OptimizerVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        """
        VAE优化器
        :param preds: Input model.reconstructions 解码生成的邻接矩阵.
        :param labels: 原始的邻接矩阵.
        :param model:
        :param num_nodes:
        :param pos_weight: 用于计算交叉熵.
        :param norm: 标准化后的邻接矩阵adj.
        """
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))

        self.cross_entropy_cost = self.cost

        # Latent loss
        self.log_lik = self.cost
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(
            tf.reduce_sum(
                1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                tf.square(tf.exp(model.z_log_std)), 1
            )
        )
        self.cost -= self.kl

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer
        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


class OptimizerCMVAE(object):
    def __init__(self, preds, labels, model, num_nodes, pos_weight, norm):
        """
        VAE优化器
        :param preds: Input model.reconstructions
        :param labels:
        :param model:
        :param num_nodes:
        :param pos_weight:
        :param norm:
        """
        preds_sub = preds
        labels_sub = labels

        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost
        # self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(1 + 2 * model.z_log_std - tf.square(model.z_mean) -
        #                                                            tf.square(tf.exp(model.z_log_std)), 1))

        # KL_divergence = tf.reduce_sum(0.5 * ((tf.square(Ex) + 1 + tf.square(En + 3 * He)) * (
        #         1 + 1 / (tf.square(En + 3 * He)))) - 1, 1)
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(tf.reduce_sum(
            0.5 * ((tf.square(model.ex) + 1 + tf.square(model.en + 3 * model.he)) * (
                    1 + 1 / (tf.square(model.en + 3 * model.he))
            )) - 1, 1
        ))

        self.cost -= self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
