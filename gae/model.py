from gae.layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg

        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

    def fit(self):
        pass

    def predict(self):
        pass


class GCNModelAE(Model):
    def __init__(self, placeholders, num_features, features_nonzero, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        # 经第一层GraphConvolutionSparse卷积，第二层GraphConvolution卷积，获得隐变量.
        self.embeddings = GraphConvolution(input_dim=FLAGS.hidden1,
                                           output_dim=FLAGS.hidden2,
                                           adj=self.adj,
                                           act=lambda x: x,
                                           dropout=self.dropout,
                                           logging=self.logging)(self.hidden1)

        self.z_mean = self.embeddings

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                   act=lambda x: x,
                                                   logging=self.logging)(self.embeddings)


class GCNModelVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.z_mean = GraphConvolution(input_dim=FLAGS.hidden1,
                                       output_dim=FLAGS.hidden2,
                                       adj=self.adj,
                                       act=lambda x: x,
                                       dropout=self.dropout,
                                       logging=self.logging)(self.hidden1)

        self.z_log_std = GraphConvolution(input_dim=FLAGS.hidden1,
                                          output_dim=FLAGS.hidden2,
                                          adj=self.adj,
                                          act=lambda x: x,
                                          dropout=self.dropout,
                                          logging=self.logging)(self.hidden1)

        self.z_std = tf.exp(self.z_log_std)

        # 测试，将z_log_std -> z_std.
        # self.z_std = GraphConvolution(input_dim=FLAGS.hidden1,
        #                               output_dim=FLAGS.hidden2,
        #                               adj=self.adj,
        #                               act=lambda x: x,
        #                               dropout=self.dropout,
        #                               logging=self.logging)(self.hidden1)

        # self.z_log_std = tf.log(self.z_std)

        '''
        z = mu + epsilon * sigma
        '''
        self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * tf.exp(self.z_log_std)
        # self.z = self.z_mean + tf.random_normal([self.n_samples, FLAGS.hidden2]) * self.z_std

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                   act=lambda x: x,
                                                   logging=self.logging)(self.z)


class GCNModelCMVAE(Model):
    def __init__(self, placeholders, num_features, num_nodes, features_nonzero, **kwargs):
        super(GCNModelCMVAE, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = num_features
        self.features_nonzero = features_nonzero
        self.n_samples = num_nodes
        self.adj = placeholders['adj']
        self.dropout = placeholders['dropout']
        self.build()

    def _build(self):
        self.hidden1 = GraphConvolutionSparse(input_dim=self.input_dim,
                                              output_dim=FLAGS.hidden1,
                                              adj=self.adj,
                                              features_nonzero=self.features_nonzero,
                                              act=tf.nn.relu,
                                              dropout=self.dropout,
                                              logging=self.logging)(self.inputs)

        self.z_ex = GraphConvolution(input_dim=FLAGS.hidden1,
                                     output_dim=FLAGS.hidden2,
                                     adj=self.adj,
                                     act=lambda x: x,
                                     dropout=self.dropout,
                                     logging=self.logging)(self.hidden1)

        # en 10* 调整.
        # self.en = 1e-6 + 10 * GraphConvolution(input_dim=FLAGS.hidden1,
        self.z_log_en = GraphConvolution(input_dim=FLAGS.hidden1,
                                         output_dim=FLAGS.hidden2,
                                         adj=self.adj,
                                         act=tf.nn.softmax,
                                         dropout=self.dropout,
                                         logging=self.logging)(self.hidden1)

        # self.he = 1e-6 + GraphConvolution(input_dim=FLAGS.hidden1,
        self.z_log_he = GraphConvolution(input_dim=FLAGS.hidden1,
                                         output_dim=FLAGS.hidden2,
                                         adj=self.adj,
                                         act=tf.nn.softmax,
                                         dropout=self.dropout,
                                         logging=self.logging)(self.hidden1)

        self.z_en = tf.exp(self.z_log_en)

        # 约束 He与En 为10倍关系.
        self.z_he = 0.1 * tf.exp(self.z_log_he)
        # self.z_he = tf.exp(self.z_log_he)

        self.z_mean = self.z_ex

        '''
        # 增加循环采样，多次采样取均值.
        # 相当于增加采样次数.
        # 实验结果表明：
        （1）模型内部多次采样，能够减小实验结果的波动范围；
        （2）能够大大减小交叉熵损失.
        （3）设置sampling_num值为1000 与 2000效果差异不大.
            设值sampling_num=500，300 epoch交叉熵能下降到0.407
            设值sampling_num=1000，300 epoch交叉熵能下降到0.404
        '''
        sampling_num = 1000
        random_sample_1 = tf.random_normal([self.n_samples, FLAGS.hidden2])
        for i in range(sampling_num - 1):
            random_sample_1 += tf.random_normal([self.n_samples, FLAGS.hidden2])
        self.sample_1 = random_sample_1 / sampling_num
        self.z_enn = self.z_en + self.sample_1 * self.z_he

        self.sample_2 = tf.random_normal([self.n_samples, FLAGS.hidden2])
        for i in range(sampling_num - 1):
            self.sample_2 += tf.random_normal([self.n_samples, FLAGS.hidden2])
        self.sample_2 = self.sample_2 / sampling_num
        self.z = self.z_ex + self.sample_2 * self.z_enn

        # self.z = self.z_ex + tf.random_normal([self.n_samples, FLAGS.hidden2]) * self.z_enn
        # for i in range(999):
        #     self.z += self.z_ex + tf.random_normal([self.n_samples, FLAGS.hidden2]) * self.z_enn
        # self.z = self.z / 1000

        '''
        # 两次采样操作.
        
        # 第一次采样
        Enn = En + epsilon * He
        '''
        # self.z_enn = self.z_en + tf.random_normal([self.n_samples, FLAGS.hidden2]) * self.z_he

        '''
        # 第二次采样
        z = Ex + epsilon * Enn
        '''
        # self.z = self.z_ex + tf.random_normal([self.n_samples, FLAGS.hidden2]) * self.z_enn

        self.reconstructions = InnerProductDecoder(input_dim=FLAGS.hidden2,
                                                   act=lambda x: x,
                                                   logging=self.logging)(self.z)
