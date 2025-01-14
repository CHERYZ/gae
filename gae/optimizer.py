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

        self.log_lik = self.cost

        # 输出hidden1.
        self.hidden1 = model.hidden1

        '''
        # Latent loss
        kl = 1/2 * (1 + 2*log_std - z_mean^2 - std^2)
        '''
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(
            tf.reduce_sum(
                1 + 2 * model.z_log_std - tf.square(model.z_mean) -
                tf.square(tf.exp(model.z_log_std)), 1
            )
        )

        # 测试，GCN输出 z_std.
        # self.kl = (0.5 / num_nodes) * tf.reduce_mean(
        #     tf.reduce_sum(
        #         1 + 2 * tf.log(model.z_std) - tf.square(model.z_mean) - tf.square(model.z_std)
        #         , 1)
        # )

        # KL(p1||p2) = -1/2 * [2*z_log_std + 1 - z_std^2 - z_mean^2]
        # 输出隐变量z，在train.py中打印输出.
        # 测试输出z_mean，z_std输出
        self.z_mean = model.z_mean
        self.z_log_std = model.z_log_std
        self.z_std = model.z_std

        # self.cost -= self.kl
        # self.kl < 0.
        self.cost = self.log_lik - self.kl

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

        '''
        model中隐变量输出测试.
        在train.debug中可打印输出结果.
        '''
        self.test_z_ex = model.z_ex
        self.test_z_log_en = model.z_log_en
        self.test_z_log_he = model.z_log_he
        self.test_z_en = model.z_en
        self.test_z_he = model.z_he

        # 交叉熵.
        self.cost = norm * tf.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(logits=preds_sub, targets=labels_sub, pos_weight=pos_weight))

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)  # Adam Optimizer

        # Latent loss
        self.log_lik = self.cost

        '''
        CMVAE中原始的KL散度输出.
        0.5 * (ex^2 + (en + 3*he)^2)
        '''
        # KL_divergence = tf.reduce_sum(
        #          0.5 * ((tf.square(Ex) + 1 + tf.square(En + 3 * He))
        #              * (1 + 1 / (tf.square(En + 3 * He)))
        #              ) - 1
        #          , 1)

        '''
        论文中 KL散度计算式：
        L = 0.5 * ((Ex1 - Ex2)^2 + sigma1^2 + sigma2^2) * (1 / sigma1^2 + 1 / sigma2^2) - 1
        sigma1 = En1 + 3*He1
        sigma2 = En2 + 3*He2
        '''

        '''
        设 Ex = 0, En = 1, He = 0;
        L = 0.5 * ( (Ex / sigma)^2 + 1 / sigma^2 + Ex^2 + sigma^2 )
        '''
        # self.sigma = model.z_en + 3 * model.z_he
        # self.kl = (0.5 / num_nodes) * tf.reduce_mean(
        #     tf.reduce_sum(
        #         tf.square(model.z_ex / self.sigma)
        #         + 1 / tf.square(self.sigma)
        #         + tf.square(model.z_ex)
        #         + tf.square(self.sigma)
        #         , 1)
        # )

        '''
        设 Ex = 0, En=1.05, He=0.1;
        L = 0.5 * {(Ex^2 + sigma^2 + 1.8225)(1 / sigma^2 + 1 / 1.35) - 2}
        '''
        self.sigma = model.z_en + 3 * model.z_he
        self.kl = (0.5 / num_nodes) * tf.reduce_mean(
            tf.reduce_sum(
                (tf.square(model.z_ex) + tf.square(self.sigma) + 1.8225)
                * (1 / tf.square(self.sigma) + 1 / 1.35)
                - 2, 1
            )
        )

        '''
        设 Ex=0, En=1.06, He=0.105
        L = 0.5 *(Ex^2 + sigma^2 + 1.890625)(1/sigma^2 + 1 / 1.375) - 1
        '''
        # self.sigma = model.z_en + 3 * model.z_he
        # self.kl = (0.5 / num_nodes) * tf.reduce_mean(
        #     tf.reduce_sum(
        #         (tf.square(model.z_ex) + tf.square(self.sigma) + 1.890625)
        #         * (1 / tf.square(self.sigma) + 1 / 1.375)
        #         - 2, 1
        #     )
        # )

        # self.cost -= self.kl
        self.cost += self.kl

        self.opt_op = self.optimizer.minimize(self.cost)
        self.grads_vars = self.optimizer.compute_gradients(self.cost)

        self.correct_prediction = tf.equal(tf.cast(tf.greater_equal(tf.sigmoid(preds_sub), 0.5), tf.int32),
                                           tf.cast(labels_sub, tf.int32))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
