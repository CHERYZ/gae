from __future__ import division
from __future__ import print_function

import time
import os
import sys

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

from gae.optimizer import OptimizerAE, OptimizerVAE, OptimizerCMVAE
from gae.input_data import load_data
from gae.model import GCNModelAE, GCNModelVAE, GCNModelCMVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 300, 'Number of epochs to train.')
# flags.DEFINE_integer('epochs', 150, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')

# flags.DEFINE_string('model', 'gcn_vae', 'Model string.')
flags.DEFINE_string('model', 'gcn_cmvae', 'Model string.')

flags.DEFINE_string('dataset', 'cora', 'Dataset string.')
# flags.DEFINE_string('dataset', 'citeseer', 'Dataset string.')
flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')

model_str = FLAGS.model
dataset_str = FLAGS.dataset


# def get_roc_score(model, edges_pos, edges_neg, emb=None):
def get_roc_score(edges_pos, edges_neg, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # Predict on test set of edges
    # 生成邻接矩阵.
    adj_rec = np.dot(emb, emb.T)
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


if __name__ == '__main__':
    p = os.path.dirname(os.path.dirname((os.path.abspath('__file__'))))
    if p not in sys.path:
        sys.path.append(p)

    '''
    多次训练求均值.
    '''
    test_roc_list = []
    test_ap_list = []
    for train_step in range(10):
        # Load data
        adj, features = load_data(dataset_str)
        # print('show data adj:\n', adj)
        # print('show data features:\n', features)

        # Store original adjacency matrix (without diagonal entries) for later
        # 存储原始邻接矩阵（没有对角线条目）以供以后使用
        adj_orig = adj
        adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
        adj_orig.eliminate_zeros()

        adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
        adj = adj_train

        if FLAGS.features == 0:
            features = sp.identity(features.shape[0])  # featureless

        # Some preprocessing
        # 归一化邻接矩阵.
        adj_norm = preprocess_graph(adj)

        # Define placeholders
        placeholders = {
            'features': tf.sparse_placeholder(tf.float32),
            'adj': tf.sparse_placeholder(tf.float32),
            'adj_orig': tf.sparse_placeholder(tf.float32),
            'dropout': tf.placeholder_with_default(0., shape=())
        }

        num_nodes = adj.shape[0]

        # 再次对features做处理.
        features = sparse_to_tuple(features.tocoo())
        num_features = features[2][1]
        features_nonzero = features[1].shape[0]

        # Create model
        model = None
        if model_str == 'gcn_ae':
            model = GCNModelAE(placeholders, num_features, features_nonzero)
        elif model_str == 'gcn_vae':
            model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)
        elif model_str == 'gcn_cmvae':
            model = GCNModelCMVAE(placeholders, num_features, num_nodes, features_nonzero)

        # 用于计算交叉熵
        pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
        norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

        # Optimizer
        with tf.name_scope('optimizer'):
            if model_str == 'gcn_ae':
                opt = OptimizerAE(preds=model.reconstructions,
                                  labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                              validate_indices=False), [-1]),
                                  pos_weight=pos_weight,
                                  norm=norm)
            elif model_str == 'gcn_vae':
                opt = OptimizerVAE(preds=model.reconstructions,
                                   labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                               validate_indices=False), [-1]),
                                   model=model, num_nodes=num_nodes,
                                   pos_weight=pos_weight,
                                   norm=norm)
            elif model_str == 'gcn_cmvae':
                opt = OptimizerCMVAE(preds=model.reconstructions,
                                     labels=tf.reshape(tf.sparse_tensor_to_dense(placeholders['adj_orig'],
                                                                                 validate_indices=False), [-1]),
                                     model=model, num_nodes=num_nodes,
                                     pos_weight=pos_weight,
                                     norm=norm)

        # Initialize session
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        train_loss_list = []
        train_acc_list = []
        train_roc_list = []
        train_ap_list = []
        val_roc_score = []

        adj_label = adj_train + sp.eye(adj_train.shape[0])
        adj_label = sparse_to_tuple(adj_label)

        # Train model
        for epoch in range(FLAGS.epochs):
            t = time.time()
            # Construct feed dictionary.
            # 构建输入数据.
            feed_dict = construct_feed_dict(adj_norm, adj_label, features, placeholders)
            feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            # Run single weight update
            # 原始输出
            if model_str == 'gcn_vae':
                outs_ori = sess.run([opt.opt_op, opt.cost, opt.accuracy,
                                     opt.log_lik, opt.kl,
                                     # 输出测试.
                                     opt.z_mean, opt.z_log_std, opt.z_std
                                     # opt.test_z_mean, opt.test_z_log_std, opt.test_z_std,
                                     # opt.hidden1], feed_dict=feed_dict)
                                     ], feed_dict=feed_dict)

                # Compute average loss
                avg_cost = outs_ori[1]
                avg_accuracy = outs_ori[2]

                # 自定义输出交叉熵部分的损失.(VAE)
                cross_entropy_cost = outs_ori[3]
                kl = outs_ori[4]

                # 输出z_mean、z_log_std、z_std.
                z_mean = outs_ori[5]
                z_log_std = outs_ori[6]
                z_std = outs_ori[7]
                # test_hidden1 = outs[10]

            elif model_str == 'gcn_cmvae':
                '''
                针对GCMVAE 的输出.
                '''
                outs_cmvae = sess.run([opt.opt_op, opt.cost, opt.accuracy,
                                       # 输出测试
                                       opt.log_lik, opt.kl,
                                       # 输出隐变量
                                       opt.test_z_ex, opt.test_z_log_en, opt.test_z_log_he,
                                       # 测试输出隐变量
                                       opt.test_z_en, opt.test_z_he
                                       # 采样
                                       # model.sample_1, model.sample_2
                                       ], feed_dict=feed_dict)

                avg_cost = outs_cmvae[1]
                avg_accuracy = outs_cmvae[2]

                # 自定义输出交叉熵部分的损失.(GCMVAE)
                cross_entropy_cost = outs_cmvae[3]
                kl = outs_cmvae[4]
                z_ex = outs_cmvae[5]
                z_log_en = outs_cmvae[6]
                z_log_he = outs_cmvae[7]
                z_en = outs_cmvae[8]
                z_he = outs_cmvae[9]
                # model_sample1 = outs_cmvae[10]
                # model_sample2 = outs_cmvae[11]

            train_loss_list.append(avg_cost)
            train_acc_list.append(avg_accuracy)

            roc_curr, ap_curr = get_roc_score(val_edges, val_edges_false)
            # roc_curr, ap_curr = get_roc_score(model=model, edges_pos=val_edges, edges_neg=val_edges_false)
            val_roc_score.append(roc_curr)

            train_roc_list.append(val_roc_score[-1])
            train_ap_list.append(ap_curr)

            print("训练次数:", train_step + 1,
                  "Epoch:", '%04d' % (epoch + 1),
                  "log_lik=", cross_entropy_cost,
                  "train_kl=", "{:.5f}".format(kl),
                  "train_loss=", "{:.5f}".format(avg_cost),
                  "train_acc=", "{:.5f}".format(avg_accuracy),
                  "val_roc=", "{:.5f}".format(val_roc_score[-1]),
                  "val_ap=", "{:.5f}".format(ap_curr),
                  "time=", "{:.5f}".format(time.time() - t))

        print("Optimization Finished!")
        roc_score, ap_score = get_roc_score(test_edges, test_edges_false)

        test_roc_list.append(roc_score)
        test_ap_list.append(ap_score)

        print("实验:", train_step + 1, 'ROC score: ' + str(roc_score))
        print("实验:", train_step + 1, 'AP score: ' + str(ap_score))

        '''
        绘制折线图
        损失变换图.
        '''
        axis = range(0, len(train_loss_list))
        plt.plot(axis, train_loss_list, '-', label="Train Loss")
        plt.show()

        # 迭代了100次，所以x的取值范围为(0，100)，然后再将每次相对应的准确率以及损失率附在x上
        # x1 = range(0, 100)
        # x2 = range(0, 100)
        # loss_draw = train_loss_list
        # ap_draw = train_ap_list
        # roc_draw = train_roc_list
        # plt.subplot(2, 1, 1)
        # plt.plot(x1, loss_draw, '-', label="Train Loss")
        # plt.title('Train Results')
        # plt.ylabel('Train Loss & Accuracy')
        # plt.legend(loc='best')
        #
        # plt.subplot(2, 1, 2)
        # plt.plot(x2, ap_draw, '.-', label="Train AP Score")
        # plt.plot(x2, roc_draw, '.-', label="Train ROC Score")
        # plt.xlabel('Epoches')
        # plt.ylabel('Train AP & ROC Score')
        # plt.legend(loc='best')
        # plt.show()

    '''
    计算ROC、AP均值.
    '''

    avg_roc = 0
    for a in test_roc_list:
        avg_roc += a
    avg_roc = avg_roc / len(test_roc_list)

    avg_ap = 0
    for b in test_ap_list:
        avg_ap += b
    avg_ap = avg_ap / len(test_ap_list)

    print('Average Test ROC score: ' + str(avg_roc))
    print('Average Test AP score: ' + str(avg_ap))

    '''
    # 绘制实验结果曲线.
    # 测试集的AUC, AP得分
    '''
    axis1 = range(0, len(test_roc_list))
    axis2 = range(0, len(test_ap_list))
    plt.subplot(2, 1, 1)
    plt.plot(axis1, test_roc_list, '-', label="AUC Score", color='r', marker='o')
    plt.title('AUC & AP Results')
    plt.ylabel('AUC Score')

    plt.subplot(2, 1, 2)
    plt.plot(axis2, test_ap_list, '-', label="AP Score", marker='^')
    plt.ylabel('AP Score')

    plt.xlabel('Train Steps')
    plt.show()


