import tensorflow as tf
import  data_utils
import logging
import os
from tensorflow.contrib.layers import *
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)


def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)


def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


def max_pool(x, ksize):
	return tf.nn.max_pool(x, ksize,
		strides=[1, 1, 1, 1], padding='VALID')
class model1(object):#BiLSTM+Attention+MLP(single) 交叉熵
    def __init__(self,config):
        self.config = config
        self.input_char = tf.placeholder(tf.int32, [None, self.config.char_sequence_length], name="input_char")
        self.input_word = tf.placeholder(tf.float32, [None, self.config.word_sequence_length,self.config.word_embedding_size], name="input_word")
        self.input_feature=tf.placeholder(tf.float32, [None, self.config.feature_size], name="input_feature")
        self.input_label = tf.placeholder(tf.float32, [None, self.config.num_classes], name="input_label")
        # self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, [], name="batch_size")
        self.real_len = tf.placeholder(tf.int32, [None], name="char_real_len")
        self.real_len_word = tf.placeholder(tf.int32, [None], name="word_real_len")

        with tf.name_scope("char_embedding"):
            _char_embeddings = tf.get_variable(shape=[self.config.char_num,
                                self.config.char_embedding_dim],dtype=tf.float32,name='char_embedding' )
            self.char_embedding = tf.nn.embedding_lookup(_char_embeddings,self.input_char, name="char_embeddings")
            self.embedded_chars_expanded = tf.expand_dims(self.char_embedding, -1)
        with tf.name_scope("char_CNN"):
            self.pooled_outputs = []
            for i, filter_size in enumerate(self.config.filter_sizes):
                # Convolution Layer
                filter_shape = [filter_size, self.config.char_embedding_dim, 1, self.config.num_filters]
                W = weight_variable(filter_shape)
                b = bias_variable([self.config.num_filters])
                conv = conv2d(self.embedded_chars_expanded, W)
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b))
                h_reshape = tf.reshape(h,
                                 shape=[self.config.batch_size, self.config.char_sequence_length - filter_size + 1, -1])
                add = tf.zeros(shape=[self.config.batch_size, filter_size - 1, self.config.num_filters])
                h_add = tf.concat([h_reshape, add], axis=1)
                if i==0:
                    self.pooled_outputs=h_add
                else:
                    self.pooled_outputs=tf.concat([self.pooled_outputs,h_add],axis=2)

            #     pooled_outputs.append(h_add)
            # num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
            # self.h_pool = tf.concat(2, [pooled_outputs[0],pooled_outputs[1],pooled_outputs[2]])
            # self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.variable_scope("char_bi-lstm"):
            if self.config.layer_num==1:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, self.char_embedding,
                        sequence_length=self.real_len, dtype=tf.float32)
                self.char_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # if self.config.dropout:
                #     self.char_outputs = tf.nn.dropout( self.char_outputs, self.config.dropout)
            else:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                stacked_fw=[]
                stacked_bw = []
                for i in range(self.config.layer_num):
                    stacked_fw.append(cell_fw)
                    stacked_bw.append(cell_bw)
                mcell_fw=tf.nn.rnn_cell.MultiRNNCell(stacked_fw)
                mcell_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_bw)
                (output_fw, output_bw), _ =tf.nn.bidirectional_dynamic_rnn(mcell_fw,mcell_bw,self.char_embedding,
                        sequence_length=self.real_len, dtype=tf.float32)
                self.char_outputs = tf.concat([output_fw, output_bw], axis=-1)

        with tf.name_scope("char_attention"):
            # W = tf.Variable(
            #     tf.truncated_normal([self.config.char_lstm_size*2, self.config.char_attention_size],
            #                         stddev=0.1), name="W")
            # b = tf.Variable(tf.random_normal([self.config.char_attention_size], stddev=0.1),
            #                 name="b")
            # u = tf.Variable(tf.random_normal([self.config.char_attention_size], stddev=0.1),
            #                 name="u")
            #
            # att = tf.tanh(
            #     tf.nn.xw_plus_b(tf.reshape(self.char_outputs, [-1, self.config.char_lstm_size*2]),
            #                     W, b),
            #     name="attention_projection"
            # )
            # logits = tf.matmul(att, tf.reshape(u, [-1, 1]), name="attention_logits")
            # attention_weights = tf.nn.softmax(
            #     tf.reshape(logits, [-1, self.config.char_sequence_length]),
            #     dim=1,
            #     name="attention_weights")
            #
            # weighted_rnn_output = tf.multiply(
            #     self.char_outputs, tf.reshape(attention_weights, [-1, self.config.char_sequence_length, 1]),
            #     name="weighted_rnn_outputs")
            # self.char_attention_outputs = tf.reduce_sum(
            #     weighted_rnn_output, 1, name="attention_outputs")
            self.char_attention_outputs = tf.reduce_max(self.char_outputs, 1)
            self.char_attention_outputs = tf.nn.dropout(
                self.char_attention_outputs, self.config.char_dropout_keep_prob,
                name="dropout")

        with tf.variable_scope("word_bi-lstm"):
            if self.config.layer_num == 1:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.input_word,
                    sequence_length=self.real_len, dtype=tf.float32)
                self.word_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # if self.config.dropout:
                #     self.word_outputs = tf.nn.dropout(self.word_outputs, self.config.dropout)
            else:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                stacked_fw = []
                stacked_bw = []
                for i in range(self.config.layer_num):
                    stacked_fw.append(cell_fw)
                    stacked_bw.append(cell_bw)
                mcell_fw = tf.nn.rnn_cell.MultiRNNCell(stacked_fw)
                mcell_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_bw)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(mcell_fw, mcell_bw, self.input_word,
                                                                            sequence_length=self.real_len_word,
                                                                            dtype=tf.float32)
                self.word_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # self.word_outputs = tf.nn.dropout(self.word_outputs, self.config.dropout)

        with tf.name_scope("word_attention"):

            # Attention mechanism
            W=tf.get_variable(name='W',shape=[self.config.word_lstm_size*2, self.config.word_attention_size],initializer=tf.glorot_uniform_initializer())
            b = tf.get_variable(name='b',shape=[self.config.word_attention_size],initializer=tf.glorot_uniform_initializer())
            u = tf.get_variable(name='u',shape=[self.config.word_attention_size],initializer=tf.glorot_uniform_initializer())

            # W = tf.Variable(
            #     tf.truncated_normal([self.config.word_lstm_size*2, self.config.word_attention_size],
            #                         stddev=0.1), name="W")
            # b = tf.Variable(tf.random_normal([self.config.word_attention_size], stddev=0.1),
            #                 name="b")
            # u = tf.Variable(tf.random_normal([self.config.word_attention_size], stddev=0.1),
            #                 name="u")

            att = tf.tanh(
                tf.nn.xw_plus_b(tf.reshape(self.word_outputs, [-1, self.config.word_lstm_size*2]),
                                W, b),
                name="attention_projection"
            )
            logits = tf.matmul(att, tf.reshape(u, [-1, 1]), name="attention_logits")
            attention_weights = tf.nn.softmax(
                tf.reshape(logits, [-1, self.config.word_sequence_length]),
                dim=1,
                name="attention_weights")

            weighted_rnn_output = tf.multiply(
                self.word_outputs, tf.reshape(attention_weights, [-1, self.config.word_sequence_length, 1]),
                name="weighted_rnn_outputs")
            self.word_attention_outputs = tf.reduce_sum(
                weighted_rnn_output, 1, name="word_attention_outputs")
            # self.word_attention_outputs = tf.reduce_max(self.word_outputs, 1)

            self.word_attention_outputs = tf.nn.dropout(
                self.word_attention_outputs, self.config.word_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("feature"):

            fc_mean, fc_var = tf.nn.moments(self.input_feature, axes=[0])
            scale = tf.Variable(tf.ones([1]))
            shift = tf.Variable(tf.zeros([1]))
            epsilon = 0.001

            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)

            mean, var = mean_var_with_update()
            input_feature = tf.nn.batch_normalization(self.input_feature, mean, var, shift, scale, epsilon)

            W = tf.get_variable(name='f_W', shape=[self.config.feature_size , self.config.feature_mlp_size],
                                initializer=tf.glorot_uniform_initializer())
            b = tf.get_variable(name='f_b', shape=[self.config.feature_mlp_size],
                                initializer=tf.glorot_uniform_initializer())

            # W = tf.Variable(
            #     tf.truncated_normal([self.config.feature_size , self.config.feature_mlp_size],
            #                         stddev=0.1), name="W")
            # b = tf.Variable(tf.random_normal([self.config.feature_mlp_size], stddev=0.1),name="b")

            # axis = list(range(2 - 1))
            # wb_mean, wb_var = tf.nn.moments(self.input_feature,axes=[1])
            # scale = tf.Variable(tf.ones([self.config.feature_size]))
            # offset = tf.Variable(tf.zeros([self.config.feature_size]))
            # variance_epsilon = 0.001
            # self.input_feature = tf.nn.batch_normalization(self.input_feature, wb_mean, wb_var, offset, scale, variance_epsilon)

            feature_output=tf.nn.xw_plus_b(input_feature, W, b)
            self.feature_output = tf.nn.dropout(
                feature_output, self.config.feature_dropout_keep_prob,
                name="dropout")
#offset一般初始化为0，scale初始化为1，
        with tf.name_scope("char_word_feature_MLP"):
            char_word_feature_outputs = tf.concat([self.char_attention_outputs,self.word_attention_outputs,self.feature_output], axis=-1)
            wb_mean, wb_var = tf.nn.moments(char_word_feature_outputs, axes=[0])
            scale = tf.Variable(tf.ones([char_word_feature_outputs.shape[1].value]))
            offset = tf.Variable(tf.zeros([char_word_feature_outputs.shape[1].value]))
            variance_epsilon = 0.001
            char_word_feature_outputs = tf.nn.batch_normalization(char_word_feature_outputs, wb_mean, wb_var, offset, scale,
                                                           variance_epsilon)

            W = tf.get_variable(name='m_W', shape=[char_word_feature_outputs.shape[1].value, self.config.MLP_size],
                                initializer=tf.glorot_uniform_initializer())
            b = tf.get_variable(name='m_b', shape=[self.config.MLP_size],
                                initializer=tf.glorot_uniform_initializer())

            # W = tf.Variable(
            #     tf.truncated_normal(
            #         [char_word_feature_outputs.shape[1].value, self.config.MLP_size], stddev=0.1
            #     ),
            #     name="W"
            # )
            # b = tf.Variable(tf.random_normal([self.config.MLP_size], stddev=0.1), name="b")
            self.mlp_output=tf.nn.xw_plus_b(char_word_feature_outputs, W, b, name="scores")
            # wb_mean, wb_var = tf.nn.moments(self.mlp_output, axes=[0])
            # scale = tf.Variable(tf.ones([self.mlp_output.shape[1].value]))
            # offset = tf.Variable(tf.zeros([self.mlp_output.shape[1].value]))
            # variance_epsilon = 0.001
            # self.mlp_output_BN = tf.nn.batch_normalization(self.mlp_output, wb_mean, wb_var, offset,
            #                                                       scale,
            #                                                       variance_epsilon)

            self.mlp_output=tf.nn.dropout(
                self.mlp_output, self.config.mlp_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("output"):

            W = tf.get_variable(name='o_W', shape=[self.mlp_output.shape[1].value, self.config.num_classes],
                                initializer=tf.glorot_uniform_initializer())
            b = tf.get_variable(name='o_b', shape=[self.config.num_classes],
                                initializer=tf.glorot_uniform_initializer())

            # W = tf.Variable(
            #     tf.truncated_normal(
            #         [self.mlp_output.shape[1].value, self.config.num_classes], stddev=0.1
            #     ),
            #     name="W"
            # )
            # b = tf.Variable(tf.random_normal([self.config.num_classes], stddev=0.1), name="b")
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.mlp_output, W, b, name="scores")

            # self.pre = tf.nn.relu(self.scores)
            # self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # CalculateMean cross-entropy loss
            # with tf.name_scope("loss"):
            #     losses = tf.pow(tf.subtract(self.pre, self.input_label), 2.0)
            #     self.loss = tf.reduce_mean(losses)  # + l1_regularizer(config.l1_reg_lambda, scope='W')(W))
                # self.loss = tf.reduce_mean(losses+l1_regularizer(config.l1_reg_lambda, scope='W')(W)) + self.config.l2_reg_lambda * l2_loss

            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.scores, labels=self.input_label
                )
                self.loss = tf.reduce_mean(losses) #+ self.config.l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_label, 1))
                self.accuracy = tf.reduce_mean(
                    tf.cast(correct_predictions, "float"),
                    name="accuracy")
                recall = tf.cast(tf.reduce_sum(
                    tf.cast(tf.div(tf.add(tf.cast(correct_predictions, dtype=tf.int64), tf.argmax(self.input_label, 1)), 2),
                            dtype=tf.int64)), dtype=tf.float32)
                true_total = tf.cast(tf.reduce_sum(tf.argmax(self.input_label, 1)), dtype=tf.float32)
                self.recall = tf.div(recall, true_total, name='recall')

            # with tf.name_scope("accuracy"):
            #     self.pre_label = tf.cast(self.pre, dtype=tf.int64)
            #     correct_predictions = tf.equal(self.pre_label, tf.cast(self.input_label, dtype=tf.int64))
            #     self.accuracy = tf.reduce_mean(
            #         tf.cast(correct_predictions, "float"),
            #         name="accuracy")
            #     recall = tf.cast(tf.reduce_sum(tf.cast(
            #         tf.div(tf.add(tf.cast(correct_predictions, dtype=tf.int64), tf.cast(self.input_label, dtype=tf.int64)),
            #                2),
            #         dtype=tf.int64)), dtype=tf.float32)
            #     true_total = tf.cast(tf.reduce_sum(self.input_label), dtype=tf.float32)
            #     self.recall = tf.div(recall, true_total, name='recall')
class model2(object):#BiLSTM+Attention+MLP(single) 交叉熵
    def __init__(self,config):
        self.config = config
        self.input_char = tf.placeholder(tf.int32, [None, self.config.char_sequence_length], name="input_char")
        self.input_word = tf.placeholder(tf.float32, [None, self.config.word_sequence_length,self.config.word_embedding_size], name="input_word")
        self.input_feature=tf.placeholder(tf.float32, [None, self.config.feature_size], name="input_feature")
        self.input_label = tf.placeholder(tf.float32, [None, self.config.num_classes], name="input_label")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, [], name="batch_size")
        self.real_len = tf.placeholder(tf.int32, [None], name="char_real_len")
        self.real_len_word = tf.placeholder(tf.int32, [None], name="word_real_len")

        with tf.name_scope("char_embedding"):
            _char_embeddings = tf.get_variable(shape=[self.config.char_num,
                                self.config.char_embedding_dim],dtype=tf.float32,name='char_embedding' )
            self.char_embedding = tf.nn.embedding_lookup(_char_embeddings,self.input_char, name="char_embeddings")

        with tf.variable_scope("char_bi-lstm"):
            if self.config.layer_num==1:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, self.char_embedding,
                        sequence_length=self.real_len, dtype=tf.float32)
                self.char_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # if self.config.dropout:
                #     self.char_outputs = tf.nn.dropout( self.char_outputs, self.config.dropout)
            else:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                stacked_fw=[]
                stacked_bw = []
                for i in range(self.config.layer_num):
                    stacked_fw.append(cell_fw)
                    stacked_bw.append(cell_bw)
                mcell_fw=tf.nn.rnn_cell.MultiRNNCell(stacked_fw)
                mcell_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_bw)
                (output_fw, output_bw), _ =tf.nn.bidirectional_dynamic_rnn(mcell_fw,mcell_bw,self.char_embedding,
                        sequence_length=self.real_len, dtype=tf.float32)
                self.char_outputs = tf.concat([output_fw, output_bw], axis=-1)

        with tf.name_scope("char_attention"):
            # Attention mechanism
            W = tf.Variable(
                tf.truncated_normal([self.config.char_lstm_size*2, self.config.char_attention_size],
                                    stddev=0.1), name="W")
            b = tf.Variable(tf.random_normal([self.config.char_attention_size], stddev=0.1),
                            name="b")
            u = tf.Variable(tf.random_normal([self.config.char_attention_size], stddev=0.1),
                            name="u")

            att = tf.tanh(
                tf.nn.xw_plus_b(tf.reshape(self.char_outputs, [-1, self.config.char_lstm_size*2]),
                                W, b),
                name="attention_projection"
            )
            logits = tf.matmul(att, tf.reshape(u, [-1, 1]), name="attention_logits")
            attention_weights = tf.nn.softmax(
                tf.reshape(logits, [-1, self.config.char_sequence_length]),
                dim=1,
                name="attention_weights")

            weighted_rnn_output = tf.multiply(
                self.char_outputs, tf.reshape(attention_weights, [-1, self.config.char_sequence_length, 1]),
                name="weighted_rnn_outputs")
            self.char_attention_outputs = tf.reduce_sum(
                weighted_rnn_output, 1, name="attention_outputs")
            self.char_attention_outputs = tf.nn.dropout(
                self.char_attention_outputs, self.config.char_dropout_keep_prob,
                name="dropout")

        with tf.variable_scope("word_bi-lstm"):
            if self.config.layer_num == 1:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.input_word,
                    sequence_length=self.real_len, dtype=tf.float32)
                self.word_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # if self.config.dropout:
                #     self.word_outputs = tf.nn.dropout(self.word_outputs, self.config.dropout)
            else:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                stacked_fw = []
                stacked_bw = []
                for i in range(self.config.layer_num):
                    stacked_fw.append(cell_fw)
                    stacked_bw.append(cell_bw)
                mcell_fw = tf.nn.rnn_cell.MultiRNNCell(stacked_fw)
                mcell_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_bw)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(mcell_fw, mcell_bw, self.input_word,
                                                                            sequence_length=self.real_len_word,
                                                                            dtype=tf.float32)
                self.word_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # self.word_outputs = tf.nn.dropout(self.word_outputs, self.config.dropout)

        with tf.name_scope("word_attention"):
            # Attention mechanism
            W = tf.Variable(
                tf.truncated_normal([self.config.word_lstm_size*2, self.config.word_attention_size],
                                    stddev=0.1), name="W")
            b = tf.Variable(tf.random_normal([self.config.word_attention_size], stddev=0.1),
                            name="b")
            u = tf.Variable(tf.random_normal([self.config.word_attention_size], stddev=0.1),
                            name="u")

            att = tf.tanh(
                tf.nn.xw_plus_b(tf.reshape(self.word_outputs, [-1, self.config.word_lstm_size*2]),
                                W, b),
                name="attention_projection"
            )
            logits = tf.matmul(att, tf.reshape(u, [-1, 1]), name="attention_logits")
            attention_weights = tf.nn.softmax(
                tf.reshape(logits, [-1, self.config.word_sequence_length]),
                dim=1,
                name="attention_weights")

            weighted_rnn_output = tf.multiply(
                self.word_outputs, tf.reshape(attention_weights, [-1, self.config.word_sequence_length, 1]),
                name="weighted_rnn_outputs")
            self.word_attention_outputs = tf.reduce_sum(
                weighted_rnn_output, 1, name="word_attention_outputs")
            self.word_attention_outputs = tf.nn.dropout(
                self.word_attention_outputs, self.config.word_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("feature"):
            W = tf.Variable(
                tf.truncated_normal([self.config.feature_size , self.config.feature_mlp_size],
                                    stddev=0.1), name="W")
            b = tf.Variable(tf.random_normal([self.config.feature_mlp_size], stddev=0.1),name="b")
            feature_output=tf.nn.xw_plus_b(self.input_feature, W, b)
            self.feature_output = tf.nn.dropout(
                feature_output, self.config.feature_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("char_word_feature_MLP"):
            char_word_feature_outputs = tf.concat([self.char_attention_outputs,self.word_attention_outputs,self.feature_output], axis=-1)
            W = tf.Variable(
                tf.truncated_normal(
                    [char_word_feature_outputs.shape[1].value, char_word_feature_outputs.shape[1].value], stddev=0.1
                ),
                name="W"
            )
            b = tf.Variable(tf.random_normal([char_word_feature_outputs.shape[1].value], stddev=0.1), name="b")
            self.mlp_output=tf.nn.xw_plus_b(char_word_feature_outputs, W, b, name="scores")
            self.mlp_output=tf.nn.dropout(
                self.mlp_output, self.config.mlp_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("output"):
            W = tf.Variable(
                tf.truncated_normal(
                    [self.mlp_output.shape[1].value, self.config.num_classes], stddev=0.1
                ),
                name="W"
            )
            b = tf.Variable(tf.random_normal([self.config.num_classes], stddev=0.1), name="b")
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.mlp_output, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.scores, labels=self.input_label
                )
                mul=tf.add(tf.multiply(tf.argmax(self.input_label, 1),self.predictions),tf.cast(tf.equal(self.predictions,tf.zeros_like(self.predictions)),dtype=tf.float32))
                l=tf.div(tf.reduce_sum(tf.log(mul)),tf.reduce_sum(self.predictions))
                self.loss = tf.reduce_mean(losses)-0.5*l #+ self.config.l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_label, 1))
                self.accuracy = tf.reduce_mean(
                    tf.cast(correct_predictions, "float"),
                    name="accuracy")
                recall = tf.cast(tf.reduce_sum(
                    tf.cast(tf.div(tf.add(tf.cast(correct_predictions, dtype=tf.int64), tf.argmax(self.input_label, 1)), 2),
                            dtype=tf.int64)), dtype=tf.float32)
                true_total = tf.cast(tf.reduce_sum(tf.argmax(self.input_label, 1)), dtype=tf.float32)
                self.recall = tf.div(recall, true_total, name='recall')
class model3(object):#BiLSTM+Attention+MLP(single) 交叉熵
    def __init__(self,config):
        self.config = config
        self.input_char = tf.placeholder(tf.int32, [None, self.config.char_sequence_length], name="input_char")
        self.input_word = tf.placeholder(tf.float32, [None, self.config.word_sequence_length,self.config.word_embedding_size], name="input_word")
        self.input_feature=tf.placeholder(tf.float32, [None, self.config.feature_size], name="input_feature")
        self.input_label = tf.placeholder(tf.float32, [None, self.config.num_classes], name="input_label")
        # self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, [], name="batch_size")
        self.real_len = tf.placeholder(tf.int32, [None], name="char_real_len")
        self.real_len_word = tf.placeholder(tf.int32, [None], name="word_real_len")

        with tf.name_scope("char_embedding"):
            _char_embeddings = tf.get_variable(shape=[self.config.char_num,
                                self.config.char_embedding_dim],dtype=tf.float32,name='char_embedding' )
            self.char_embedding = tf.nn.embedding_lookup(_char_embeddings,self.input_char, name="char_embeddings")

        with tf.variable_scope("char_bi-lstm"):
            if self.config.layer_num==1:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, self.char_embedding,
                        sequence_length=self.real_len, dtype=tf.float32)
                self.char_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # if self.config.dropout:
                #     self.char_outputs = tf.nn.dropout( self.char_outputs, self.config.dropout)
            else:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                stacked_fw=[]
                stacked_bw = []
                for i in range(self.config.layer_num):
                    stacked_fw.append(cell_fw)
                    stacked_bw.append(cell_bw)
                mcell_fw=tf.nn.rnn_cell.MultiRNNCell(stacked_fw)
                mcell_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_bw)
                (output_fw, output_bw), _ =tf.nn.bidirectional_dynamic_rnn(mcell_fw,mcell_bw,self.char_embedding,
                        sequence_length=self.real_len, dtype=tf.float32)
                self.char_outputs = tf.concat([output_fw, output_bw], axis=-1)

        with tf.name_scope("char_attention"):
            # W = tf.Variable(
            #     tf.truncated_normal([self.config.char_lstm_size*2, self.config.char_attention_size],
            #                         stddev=0.1), name="W")
            # b = tf.Variable(tf.random_normal([self.config.char_attention_size], stddev=0.1),
            #                 name="b")
            # u = tf.Variable(tf.random_normal([self.config.char_attention_size], stddev=0.1),
            #                 name="u")
            #
            # att = tf.tanh(
            #     tf.nn.xw_plus_b(tf.reshape(self.char_outputs, [-1, self.config.char_lstm_size*2]),
            #                     W, b),
            #     name="attention_projection"
            # )
            # logits = tf.matmul(att, tf.reshape(u, [-1, 1]), name="attention_logits")
            # attention_weights = tf.nn.softmax(
            #     tf.reshape(logits, [-1, self.config.char_sequence_length]),
            #     dim=1,
            #     name="attention_weights")
            #
            # weighted_rnn_output = tf.multiply(
            #     self.char_outputs, tf.reshape(attention_weights, [-1, self.config.char_sequence_length, 1]),
            #     name="weighted_rnn_outputs")
            # self.char_attention_outputs = tf.reduce_sum(
            #     weighted_rnn_output, 1, name="attention_outputs")
            self.char_attention_outputs = tf.reduce_max(self.char_outputs, 1)
            self.char_attention_outputs = tf.nn.dropout(
                self.char_attention_outputs, self.config.char_dropout_keep_prob,
                name="dropout")

        with tf.variable_scope("word_bi-lstm"):
            if self.config.layer_num == 1:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.input_word,
                    sequence_length=self.real_len, dtype=tf.float32)
                self.word_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # if self.config.dropout:
                #     self.word_outputs = tf.nn.dropout(self.word_outputs, self.config.dropout)
            else:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                stacked_fw = []
                stacked_bw = []
                for i in range(self.config.layer_num):
                    stacked_fw.append(cell_fw)
                    stacked_bw.append(cell_bw)
                mcell_fw = tf.nn.rnn_cell.MultiRNNCell(stacked_fw)
                mcell_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_bw)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(mcell_fw, mcell_bw, self.input_word,
                                                                            sequence_length=self.real_len_word,
                                                                            dtype=tf.float32)
                self.word_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # self.word_outputs = tf.nn.dropout(self.word_outputs, self.config.dropout)

        with tf.name_scope("word_attention"):
            # Attention mechanism
            W = tf.Variable(
                tf.truncated_normal([self.config.word_lstm_size*2, self.config.word_attention_size],
                                    stddev=0.1), name="W")
            b = tf.Variable(tf.random_normal([self.config.word_attention_size], stddev=0.1),
                            name="b")
            u = tf.Variable(tf.random_normal([self.config.word_attention_size], stddev=0.1),
                            name="u")

            att = tf.tanh(
                tf.nn.xw_plus_b(tf.reshape(self.word_outputs, [-1, self.config.word_lstm_size*2]),
                                W, b),
                name="attention_projection"
            )
            logits = tf.matmul(att, tf.reshape(u, [-1, 1]), name="attention_logits")
            attention_weights = tf.nn.softmax(
                tf.reshape(logits, [-1, self.config.word_sequence_length]),
                dim=1,
                name="attention_weights")

            weighted_rnn_output = tf.multiply(
                self.word_outputs, tf.reshape(attention_weights, [-1, self.config.word_sequence_length, 1]),
                name="weighted_rnn_outputs")
            self.word_attention_outputs = tf.reduce_sum(
                weighted_rnn_output, 1, name="word_attention_outputs")
            # self.word_attention_outputs = tf.reduce_max(self.word_outputs, 1)

            self.word_attention_outputs = tf.nn.dropout(
                self.word_attention_outputs, self.config.word_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("feature"):

            fc_mean, fc_var = tf.nn.moments(self.input_feature, axes=[0])
            scale = tf.Variable(tf.ones([1]))
            shift = tf.Variable(tf.zeros([1]))
            epsilon = 0.001

            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)

            mean, var = mean_var_with_update()
            input_feature = tf.nn.batch_normalization(self.input_feature, mean, var, shift, scale, epsilon)

            W = tf.Variable(
                tf.truncated_normal([self.config.feature_size , self.config.feature_mlp_size],
                                    stddev=0.1), name="W")
            b = tf.Variable(tf.random_normal([self.config.feature_mlp_size], stddev=0.1),name="b")

            # axis = list(range(2 - 1))
            # wb_mean, wb_var = tf.nn.moments(self.input_feature,axes=[1])
            # scale = tf.Variable(tf.ones([self.config.feature_size]))
            # offset = tf.Variable(tf.zeros([self.config.feature_size]))
            # variance_epsilon = 0.001
            # self.input_feature = tf.nn.batch_normalization(self.input_feature, wb_mean, wb_var, offset, scale, variance_epsilon)

            feature_output=tf.nn.xw_plus_b(input_feature, W, b)
            self.feature_output = tf.nn.dropout(
                feature_output, self.config.feature_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("char_word_feature_MLP"):
            char_word_feature_outputs = tf.concat([self.char_attention_outputs,self.word_attention_outputs,self.feature_output], axis=-1)
            wb_mean, wb_var = tf.nn.moments(char_word_feature_outputs, axes=[0])
            scale = tf.Variable(tf.ones([char_word_feature_outputs.shape[1].value]))
            offset = tf.Variable(tf.zeros([char_word_feature_outputs.shape[1].value]))
            variance_epsilon = 0.001
            char_word_feature_outputs = tf.nn.batch_normalization(char_word_feature_outputs, wb_mean, wb_var, offset, scale,
                                                           variance_epsilon)

            W = tf.Variable(
                tf.truncated_normal(
                    [char_word_feature_outputs.shape[1].value, self.config.MLP_size], stddev=0.1
                ),
                name="W"
            )
            b = tf.Variable(tf.random_normal([self.config.MLP_size], stddev=0.1), name="b")
            self.mlp_output=tf.nn.xw_plus_b(char_word_feature_outputs, W, b, name="scores")
            # wb_mean, wb_var = tf.nn.moments(self.mlp_output, axes=[0])
            # scale = tf.Variable(tf.ones([self.mlp_output.shape[1].value]))
            # offset = tf.Variable(tf.zeros([self.mlp_output.shape[1].value]))
            # variance_epsilon = 0.001
            # self.mlp_output_BN = tf.nn.batch_normalization(self.mlp_output, wb_mean, wb_var, offset,
            #                                                       scale,
            #                                                       variance_epsilon)

            self.mlp_output=tf.nn.dropout(
                self.mlp_output, self.config.mlp_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("output"):
            W = tf.Variable(
                tf.truncated_normal(
                    [self.mlp_output.shape[1].value, self.config.num_classes], stddev=0.1
                ),
                name="W"
            )
            b = tf.Variable(tf.random_normal([self.config.num_classes], stddev=0.1), name="b")
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.mlp_output, W, b, name="scores")

            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.scores, labels=self.input_label
                )
                mul = tf.add(tf.cast(tf.multiply(tf.argmax(self.input_label, 1), self.predictions),dtype=tf.float32),
                             tf.cast(tf.equal(self.predictions, tf.zeros_like(self.predictions)), dtype=tf.float32))
                l = tf.div(tf.reduce_sum(tf.log(mul)), tf.cast(tf.reduce_sum(self.predictions),dtype=tf.float32))
                self.loss = tf.reduce_mean(losses) - 0.5 * l  # + self.config.l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_label, 1))
                self.accuracy = tf.reduce_mean(
                    tf.cast(correct_predictions, "float"),
                    name="accuracy")
                recall = tf.cast(tf.reduce_sum(
                    tf.cast(tf.div(tf.add(tf.cast(correct_predictions, dtype=tf.int64), tf.argmax(self.input_label, 1)), 2),
                            dtype=tf.int64)), dtype=tf.float32)
                true_total = tf.cast(tf.reduce_sum(tf.argmax(self.input_label, 1)), dtype=tf.float32)
                self.recall = tf.div(recall, true_total, name='recall')
class model4(object):#BiLSTM+Attention+MLP(single) 交叉熵
    def __init__(self,config):
        self.config = config
        self.input_char = tf.placeholder(tf.int32, [None, self.config.char_sequence_length], name="input_char")
        self.input_word = tf.placeholder(tf.float32, [None, self.config.word_sequence_length,self.config.word_embedding_size], name="input_word")
        self.input_feature=tf.placeholder(tf.float32, [None, self.config.feature_size], name="input_feature")
        self.input_label = tf.placeholder(tf.float32, [None, self.config.num_classes], name="input_label")
        # self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, [], name="batch_size")
        self.real_len = tf.placeholder(tf.int32, [None], name="char_real_len")
        self.real_len_word = tf.placeholder(tf.int32, [None], name="word_real_len")

        with tf.name_scope("char_embedding"):
            _char_embeddings = tf.get_variable(shape=[self.config.char_num,
                                self.config.char_embedding_dim],dtype=tf.float32,name='char_embedding' )
            self.char_embedding = tf.nn.embedding_lookup(_char_embeddings,self.input_char, name="char_embeddings")
            self.embedded_chars_expanded = tf.expand_dims(self.char_embedding, -1)
        with tf.name_scope("char_CNN"):
            self.CNN_outputs = []
            for i, filter_size in enumerate(self.config.filter_sizes):
                # Convolution Layer
                filter_shape = [filter_size, self.config.char_embedding_dim, 1, self.config.num_filters]
                W = weight_variable(filter_shape)
                b = bias_variable([self.config.num_filters])
                conv = conv2d(self.embedded_chars_expanded, W)
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b))
                h_reshape = tf.reshape(h,
                                       shape=[-1,
                                              self.config.char_sequence_length - filter_size + 1, self.config.num_filters])
                add = tf.zeros(shape=[self.batch_size, filter_size - 1, self.config.num_filters])
                h_add = tf.concat([h_reshape, add], axis=1)
                if i == 0:
                    self.CNN_outputs = h_add
                else:
                    self.CNN_outputs = tf.concat([self.CNN_outputs, h_add], axis=2)

            #     pooled_outputs.append(h_add)
            # num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
            # self.h_pool = tf.concat(2, [pooled_outputs[0],pooled_outputs[1],pooled_outputs[2]])
            # self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        with tf.variable_scope("char_bi-lstm"):
            if self.config.layer_num==1:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.num_filters*len(self.config.filter_sizes))
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.num_filters*len(self.config.filter_sizes))
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, self.CNN_outputs,
                        sequence_length=self.real_len, dtype=tf.float32)
                self.char_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # if self.config.dropout:
                #     self.char_outputs = tf.nn.dropout( self.char_outputs, self.config.dropout)
            else:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.num_filters*len(self.config.filter_sizes))
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.num_filters*len(self.config.filter_sizes))
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                stacked_fw=[]
                stacked_bw = []
                for i in range(self.config.layer_num):
                    stacked_fw.append(cell_fw)
                    stacked_bw.append(cell_bw)
                mcell_fw=tf.nn.rnn_cell.MultiRNNCell(stacked_fw)
                mcell_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_bw)
                (output_fw, output_bw), _ =tf.nn.bidirectional_dynamic_rnn(mcell_fw,mcell_bw,self.CNN_outputs,
                        sequence_length=self.real_len, dtype=tf.float32)
                self.char_outputs = tf.concat([output_fw, output_bw], axis=-1)

        with tf.name_scope("char_attention"):
            # W = tf.Variable(
            #     tf.truncated_normal([self.config.char_lstm_size*2, self.config.char_attention_size],
            #                         stddev=0.1), name="W")
            # b = tf.Variable(tf.random_normal([self.config.char_attention_size], stddev=0.1),
            #                 name="b")
            # u = tf.Variable(tf.random_normal([self.config.char_attention_size], stddev=0.1),
            #                 name="u")
            #
            # att = tf.tanh(
            #     tf.nn.xw_plus_b(tf.reshape(self.char_outputs, [-1, self.config.char_lstm_size*2]),
            #                     W, b),
            #     name="attention_projection"
            # )
            # logits = tf.matmul(att, tf.reshape(u, [-1, 1]), name="attention_logits")
            # attention_weights = tf.nn.softmax(
            #     tf.reshape(logits, [-1, self.config.char_sequence_length]),
            #     dim=1,
            #     name="attention_weights")
            #
            # weighted_rnn_output = tf.multiply(
            #     self.char_outputs, tf.reshape(attention_weights, [-1, self.config.char_sequence_length, 1]),
            #     name="weighted_rnn_outputs")
            # self.char_attention_outputs = tf.reduce_sum(
            #     weighted_rnn_output, 1, name="attention_outputs")
            self.char_attention_outputs = tf.reduce_max(self.char_outputs, 1)
            self.char_attention_outputs = tf.nn.dropout(
                self.char_attention_outputs, self.config.char_dropout_keep_prob,
                name="dropout")

        with tf.variable_scope("word_bi-lstm"):
            if self.config.layer_num == 1:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.input_word,
                    sequence_length=self.real_len, dtype=tf.float32)
                self.word_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # if self.config.dropout:
                #     self.word_outputs = tf.nn.dropout(self.word_outputs, self.config.dropout)
            else:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                stacked_fw = []
                stacked_bw = []
                for i in range(self.config.layer_num):
                    stacked_fw.append(cell_fw)
                    stacked_bw.append(cell_bw)
                mcell_fw = tf.nn.rnn_cell.MultiRNNCell(stacked_fw)
                mcell_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_bw)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(mcell_fw, mcell_bw, self.input_word,
                                                                            sequence_length=self.real_len_word,
                                                                            dtype=tf.float32)
                self.word_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # self.word_outputs = tf.nn.dropout(self.word_outputs, self.config.dropout)

        with tf.name_scope("word_attention"):
            # Attention mechanism
            W = tf.Variable(
                tf.truncated_normal([self.config.word_lstm_size*2, self.config.word_attention_size],
                                    stddev=0.1), name="W")
            b = tf.Variable(tf.random_normal([self.config.word_attention_size], stddev=0.1),
                            name="b")
            u = tf.Variable(tf.random_normal([self.config.word_attention_size], stddev=0.1),
                            name="u")

            att = tf.tanh(
                tf.nn.xw_plus_b(tf.reshape(self.word_outputs, [-1, self.config.word_lstm_size*2]),
                                W, b),
                name="attention_projection"
            )
            logits = tf.matmul(att, tf.reshape(u, [-1, 1]), name="attention_logits")
            attention_weights = tf.nn.softmax(
                tf.reshape(logits, [-1, self.config.word_sequence_length]),
                dim=1,
                name="attention_weights")

            weighted_rnn_output = tf.multiply(
                self.word_outputs, tf.reshape(attention_weights, [-1, self.config.word_sequence_length, 1]),
                name="weighted_rnn_outputs")
            self.word_attention_outputs = tf.reduce_sum(
                weighted_rnn_output, 1, name="word_attention_outputs")
            # self.word_attention_outputs = tf.reduce_max(self.word_outputs, 1)

            self.word_attention_outputs = tf.nn.dropout(
                self.word_attention_outputs, self.config.word_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("feature"):

            fc_mean, fc_var = tf.nn.moments(self.input_feature, axes=[0])
            scale = tf.Variable(tf.ones([1]))
            shift = tf.Variable(tf.zeros([1]))
            epsilon = 0.001

            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)

            mean, var = mean_var_with_update()
            input_feature = tf.nn.batch_normalization(self.input_feature, mean, var, shift, scale, epsilon)

            W = tf.Variable(
                tf.truncated_normal([self.config.feature_size , self.config.feature_mlp_size],
                                    stddev=0.1), name="W")
            b = tf.Variable(tf.random_normal([self.config.feature_mlp_size], stddev=0.1),name="b")

            # axis = list(range(2 - 1))
            # wb_mean, wb_var = tf.nn.moments(self.input_feature,axes=[1])
            # scale = tf.Variable(tf.ones([self.config.feature_size]))
            # offset = tf.Variable(tf.zeros([self.config.feature_size]))
            # variance_epsilon = 0.001
            # self.input_feature = tf.nn.batch_normalization(self.input_feature, wb_mean, wb_var, offset, scale, variance_epsilon)

            feature_output=tf.nn.xw_plus_b(input_feature, W, b)
            self.feature_output = tf.nn.dropout(
                feature_output, self.config.feature_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("char_word_feature_MLP"):
            char_word_feature_outputs = tf.concat([self.char_attention_outputs,self.word_attention_outputs,self.feature_output], axis=-1)
            wb_mean, wb_var = tf.nn.moments(char_word_feature_outputs, axes=[0])
            scale = tf.Variable(tf.ones([char_word_feature_outputs.shape[1].value]))
            offset = tf.Variable(tf.zeros([char_word_feature_outputs.shape[1].value]))
            variance_epsilon = 0.001
            char_word_feature_outputs = tf.nn.batch_normalization(char_word_feature_outputs, wb_mean, wb_var, offset, scale,
                                                           variance_epsilon)

            W = tf.Variable(
                tf.truncated_normal(
                    [char_word_feature_outputs.shape[1].value, self.config.MLP_size], stddev=0.1
                ),
                name="W"
            )
            b = tf.Variable(tf.random_normal([self.config.MLP_size], stddev=0.1), name="b")
            self.mlp_output=tf.nn.xw_plus_b(char_word_feature_outputs, W, b, name="scores")
            # wb_mean, wb_var = tf.nn.moments(self.mlp_output, axes=[0])
            # scale = tf.Variable(tf.ones([self.mlp_output.shape[1].value]))
            # offset = tf.Variable(tf.zeros([self.mlp_output.shape[1].value]))
            # variance_epsilon = 0.001
            # self.mlp_output_BN = tf.nn.batch_normalization(self.mlp_output, wb_mean, wb_var, offset,
            #                                                       scale,
            #                                                       variance_epsilon)

            self.mlp_output=tf.nn.dropout(
                self.mlp_output, self.config.mlp_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("output"):
            W = tf.Variable(
                tf.truncated_normal(
                    [self.mlp_output.shape[1].value, self.config.num_classes], stddev=0.1
                ),
                name="W"
            )
            b = tf.Variable(tf.random_normal([self.config.num_classes], stddev=0.1), name="b")
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.mlp_output, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.scores, labels=self.input_label
                )
                self.loss = tf.reduce_mean(losses) #+ self.config.l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_label, 1))
                self.accuracy = tf.reduce_mean(
                    tf.cast(correct_predictions, "float"),
                    name="accuracy")
                recall = tf.cast(tf.reduce_sum(
                    tf.cast(tf.div(tf.add(tf.cast(correct_predictions, dtype=tf.int64), tf.argmax(self.input_label, 1)), 2),
                            dtype=tf.int64)), dtype=tf.float32)
                true_total = tf.cast(tf.reduce_sum(tf.argmax(self.input_label, 1)), dtype=tf.float32)
                self.recall = tf.div(recall, true_total, name='recall')
class model_dev1(object):#BiLSTM+Attention+MLP(single) 交叉熵
    def __init__(self,config):
        self.config = config
        self.input_char = tf.placeholder(tf.int32, [None, self.config.char_sequence_length], name="input_char")
        self.input_word = tf.placeholder(tf.float32, [None, self.config.word_sequence_length,self.config.word_embedding_size], name="input_word")
        self.input_feature=tf.placeholder(tf.float32, [None, self.config.feature_size], name="input_feature")
        self.input_label = tf.placeholder(tf.float32, [None, self.config.num_classes], name="input_label")
        # self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, [], name="batch_size")
        self.real_len = tf.placeholder(tf.int32, [None], name="char_real_len")
        self.real_len_word = tf.placeholder(tf.int32, [None], name="word_real_len")

        with tf.name_scope("char_embedding"):
            _char_embeddings = tf.get_variable(shape=[self.config.char_num,
                                self.config.char_embedding_dim],dtype=tf.float32,name='char_embedding' )
            self.char_embedding = tf.nn.embedding_lookup(_char_embeddings,self.input_char, name="char_embeddings")

        with tf.variable_scope("char_bi-lstm"):
            if self.config.layer_num==1:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, self.char_embedding,
                        sequence_length=self.real_len, dtype=tf.float32)
                self.char_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # if self.config.dropout:
                #     self.char_outputs = tf.nn.dropout( self.char_outputs, self.config.dropout)
            else:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                stacked_fw=[]
                stacked_bw = []
                for i in range(self.config.layer_num):
                    stacked_fw.append(cell_fw)
                    stacked_bw.append(cell_bw)
                mcell_fw=tf.nn.rnn_cell.MultiRNNCell(stacked_fw)
                mcell_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_bw)
                (output_fw, output_bw), _ =tf.nn.bidirectional_dynamic_rnn(mcell_fw,mcell_bw,self.char_embedding,
                        sequence_length=self.real_len, dtype=tf.float32)
                self.char_outputs = tf.concat([output_fw, output_bw], axis=-1)

        with tf.name_scope("char_attention"):
            # W = tf.Variable(
            #     tf.truncated_normal([self.config.char_lstm_size*2, self.config.char_attention_size],
            #                         stddev=0.1), name="W")
            # b = tf.Variable(tf.random_normal([self.config.char_attention_size], stddev=0.1),
            #                 name="b")
            # u = tf.Variable(tf.random_normal([self.config.char_attention_size], stddev=0.1),
            #                 name="u")
            #
            # att = tf.tanh(
            #     tf.nn.xw_plus_b(tf.reshape(self.char_outputs, [-1, self.config.char_lstm_size*2]),
            #                     W, b),
            #     name="attention_projection"
            # )
            # logits = tf.matmul(att, tf.reshape(u, [-1, 1]), name="attention_logits")
            # attention_weights = tf.nn.softmax(
            #     tf.reshape(logits, [-1, self.config.char_sequence_length]),
            #     dim=1,
            #     name="attention_weights")
            #
            # weighted_rnn_output = tf.multiply(
            #     self.char_outputs, tf.reshape(attention_weights, [-1, self.config.char_sequence_length, 1]),
            #     name="weighted_rnn_outputs")
            # self.char_attention_outputs = tf.reduce_sum(
            #     weighted_rnn_output, 1, name="attention_outputs")
            self.char_attention_outputs = tf.reduce_max(self.char_outputs, 1)
            self.char_attention_outputs = tf.nn.dropout(
                self.char_attention_outputs, self.config.char_dropout_keep_prob,
                name="dropout")

        with tf.variable_scope("word_bi-lstm"):
            if self.config.layer_num == 1:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.input_word,
                    sequence_length=self.real_len, dtype=tf.float32)
                self.word_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # if self.config.dropout:
                #     self.word_outputs = tf.nn.dropout(self.word_outputs, self.config.dropout)
            else:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                stacked_fw = []
                stacked_bw = []
                for i in range(self.config.layer_num):
                    stacked_fw.append(cell_fw)
                    stacked_bw.append(cell_bw)
                mcell_fw = tf.nn.rnn_cell.MultiRNNCell(stacked_fw)
                mcell_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_bw)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(mcell_fw, mcell_bw, self.input_word,
                                                                            sequence_length=self.real_len_word,
                                                                            dtype=tf.float32)
                self.word_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # self.word_outputs = tf.nn.dropout(self.word_outputs, self.config.dropout)

        with tf.name_scope("word_attention"):
            # Attention mechanism
            W = tf.Variable(
                tf.truncated_normal([self.config.word_lstm_size*2, self.config.word_attention_size],
                                    stddev=0.1), name="W")
            b = tf.Variable(tf.random_normal([self.config.word_attention_size], stddev=0.1),
                            name="b")
            u = tf.Variable(tf.random_normal([self.config.word_attention_size], stddev=0.1),
                            name="u")

            att = tf.tanh(
                tf.nn.xw_plus_b(tf.reshape(self.word_outputs, [-1, self.config.word_lstm_size*2]),
                                W, b),
                name="attention_projection"
            )
            logits = tf.matmul(att, tf.reshape(u, [-1, 1]), name="attention_logits")
            attention_weights = tf.nn.softmax(
                tf.reshape(logits, [-1, self.config.word_sequence_length]),
                dim=1,
                name="attention_weights")

            weighted_rnn_output = tf.multiply(
                self.word_outputs, tf.reshape(attention_weights, [-1, self.config.word_sequence_length, 1]),
                name="weighted_rnn_outputs")
            self.word_attention_outputs = tf.reduce_sum(
                weighted_rnn_output, 1, name="word_attention_outputs")
            # self.word_attention_outputs = tf.reduce_max(self.word_outputs, 1)

            self.word_attention_outputs = tf.nn.dropout(
                self.word_attention_outputs, self.config.word_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("feature"):

            fc_mean, fc_var = tf.nn.moments(self.input_feature, axes=[0])
            scale = tf.Variable(tf.ones([1]))
            shift = tf.Variable(tf.zeros([1]))
            epsilon = 0.001

            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)

            mean, var = mean_var_with_update()
            input_feature = tf.nn.batch_normalization(self.input_feature, mean, var, shift, scale, epsilon)

            W = tf.Variable(
                tf.truncated_normal([self.config.feature_size , self.config.feature_mlp_size],
                                    stddev=0.1), name="W")
            b = tf.Variable(tf.random_normal([self.config.feature_mlp_size], stddev=0.1),name="b")

            # axis = list(range(2 - 1))
            # wb_mean, wb_var = tf.nn.moments(self.input_feature,axes=[1])
            # scale = tf.Variable(tf.ones([self.config.feature_size]))
            # offset = tf.Variable(tf.zeros([self.config.feature_size]))
            # variance_epsilon = 0.001
            # self.input_feature = tf.nn.batch_normalization(self.input_feature, wb_mean, wb_var, offset, scale, variance_epsilon)

            feature_output=tf.nn.xw_plus_b(input_feature, W, b)
            self.feature_output = tf.nn.dropout(
                feature_output, self.config.feature_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("char_word_feature_MLP"):
            char_word_feature_outputs = tf.concat([self.char_attention_outputs,self.word_attention_outputs,self.feature_output], axis=-1)
            wb_mean, wb_var = tf.nn.moments(char_word_feature_outputs, axes=[0])
            scale = tf.Variable(tf.ones([char_word_feature_outputs.shape[1].value]))
            offset = tf.Variable(tf.zeros([char_word_feature_outputs.shape[1].value]))
            variance_epsilon = 0.001
            char_word_feature_outputs = tf.nn.batch_normalization(char_word_feature_outputs, wb_mean, wb_var, offset, scale,
                                                           variance_epsilon)

            W = tf.Variable(
                tf.truncated_normal(
                    [char_word_feature_outputs.shape[1].value, self.config.MLP_size], stddev=0.1
                ),
                name="W"
            )
            b = tf.Variable(tf.random_normal([self.config.MLP_size], stddev=0.1), name="b")
            self.mlp_output=tf.nn.xw_plus_b(char_word_feature_outputs, W, b, name="scores")
            # wb_mean, wb_var = tf.nn.moments(self.mlp_output, axes=[0])
            # scale = tf.Variable(tf.ones([self.mlp_output.shape[1].value]))
            # offset = tf.Variable(tf.zeros([self.mlp_output.shape[1].value]))
            # variance_epsilon = 0.001
            # self.mlp_output_BN = tf.nn.batch_normalization(self.mlp_output, wb_mean, wb_var, offset,
            #                                                       scale,
            #                                                       variance_epsilon)

            self.mlp_output=tf.nn.dropout(
                self.mlp_output, self.config.mlp_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("output"):
            W = tf.Variable(
                tf.truncated_normal(
                    [self.mlp_output.shape[1].value, self.config.num_classes], stddev=0.1
                ),
                name="W"
            )
            b = tf.Variable(tf.random_normal([self.config.num_classes], stddev=0.1), name="b")
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.mlp_output, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.scores, labels=self.input_label
                )
                self.loss = tf.reduce_mean(losses) #+ self.config.l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_label, 1))
                self.accuracy = tf.reduce_mean(
                    tf.cast(correct_predictions, "float"),
                    name="accuracy")
                recall = tf.cast(tf.reduce_sum(
                    tf.cast(tf.div(tf.add(tf.cast(correct_predictions, dtype=tf.int64), tf.argmax(self.input_label, 1)), 2),
                            dtype=tf.int64)), dtype=tf.float32)
                true_total = tf.cast(tf.reduce_sum(tf.argmax(self.input_label, 1)), dtype=tf.float32)
                self.recall = tf.div(recall, true_total, name='recall')
class model5(object):#BiLSTM+Attention+MLP(single) 交叉熵
    def __init__(self,config):
        self.config = config
        self.input_char = tf.placeholder(tf.int32, [None, self.config.char_sequence_length], name="input_char")
        self.input_word = tf.placeholder(tf.float32, [None, self.config.word_sequence_length,self.config.word_embedding_size], name="input_word")
        self.input_feature=tf.placeholder(tf.float32, [None, self.config.feature_size], name="input_feature")
        self.input_label = tf.placeholder(tf.float32, [None, self.config.num_classes], name="input_label")
        # self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, [], name="batch_size")
        self.real_len = tf.placeholder(tf.int32, [None], name="char_real_len")
        self.real_len_word = tf.placeholder(tf.int32, [None], name="word_real_len")

        with tf.name_scope("char_embedding"):
            _char_embeddings = tf.get_variable(shape=[self.config.char_num,
                                self.config.char_embedding_dim],dtype=tf.float32,name='char_embedding' )
            self.char_embedding = tf.nn.embedding_lookup(_char_embeddings,self.input_char, name="char_embeddings")

        with tf.variable_scope("char_bi-lstm"):
            if self.config.layer_num==1:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, self.char_embedding,
                        sequence_length=self.real_len, dtype=tf.float32)
                self.char_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # if self.config.dropout:
                #     self.char_outputs = tf.nn.dropout( self.char_outputs, self.config.dropout)
            else:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                stacked_fw=[]
                stacked_bw = []
                state_fw=tf.get_variable(shape=[self.config.batch_size,self.config.char_lstm_size],initializer=tf.glorot_uniform_initializer())
                state_bw = tf.get_variable(shape=[self.config.batch_size, self.config.char_lstm_size],
                                           initializer=tf.glorot_uniform_initializer())
                for i in range(self.config.layer_num):
                    stacked_fw.append(cell_fw)
                    stacked_bw.append(cell_bw)
                mcell_fw=tf.nn.rnn_cell.MultiRNNCell(stacked_fw)
                mcell_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_bw)
                (output_fw, output_bw), _ =tf.nn.bidirectional_dynamic_rnn(mcell_fw,mcell_bw,self.char_embedding,
                        sequence_length=self.real_len, initial_state_fw=state_fw,initial_state_bw=state_bw,dtype=tf.float32)
                self.char_outputs = tf.concat([output_fw, output_bw], axis=-1)

        with tf.name_scope("char_attention"):
            # W = tf.Variable(
            #     tf.truncated_normal([self.config.char_lstm_size*2, self.config.char_attention_size],
            #                         stddev=0.1), name="W")
            # b = tf.Variable(tf.random_normal([self.config.char_attention_size], stddev=0.1),
            #                 name="b")
            # u = tf.Variable(tf.random_normal([self.config.char_attention_size], stddev=0.1),
            #                 name="u")
            #
            # att = tf.tanh(
            #     tf.nn.xw_plus_b(tf.reshape(self.char_outputs, [-1, self.config.char_lstm_size*2]),
            #                     W, b),
            #     name="attention_projection"
            # )
            # logits = tf.matmul(att, tf.reshape(u, [-1, 1]), name="attention_logits")
            # attention_weights = tf.nn.softmax(
            #     tf.reshape(logits, [-1, self.config.char_sequence_length]),
            #     dim=1,
            #     name="attention_weights")
            #
            # weighted_rnn_output = tf.multiply(
            #     self.char_outputs, tf.reshape(attention_weights, [-1, self.config.char_sequence_length, 1]),
            #     name="weighted_rnn_outputs")
            # self.char_attention_outputs = tf.reduce_sum(
            #     weighted_rnn_output, 1, name="attention_outputs")
            self.char_attention_outputs = tf.reduce_max(self.char_outputs, 1)
            self.char_attention_outputs = tf.nn.dropout(
                self.char_attention_outputs, self.config.char_dropout_keep_prob,
                name="dropout")

        with tf.variable_scope("word_bi-lstm"):
            if self.config.layer_num == 1:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.input_word,
                    sequence_length=self.real_len, dtype=tf.float32)
                self.word_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # if self.config.dropout:
                #     self.word_outputs = tf.nn.dropout(self.word_outputs, self.config.dropout)
            else:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                stacked_fw = []
                stacked_bw = []
                for i in range(self.config.layer_num):
                    stacked_fw.append(cell_fw)
                    stacked_bw.append(cell_bw)
                mcell_fw = tf.nn.rnn_cell.MultiRNNCell(stacked_fw)
                mcell_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_bw)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(mcell_fw, mcell_bw, self.input_word,
                                                                            sequence_length=self.real_len_word,
                                                                            dtype=tf.float32)
                self.word_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # self.word_outputs = tf.nn.dropout(self.word_outputs, self.config.dropout)

        with tf.name_scope("word_attention"):

            # Attention mechanism
            W=tf.get_variable(name='W',shape=[self.config.word_lstm_size*2, self.config.word_attention_size],initializer=tf.glorot_uniform_initializer())
            b = tf.get_variable(name='b',shape=[self.config.word_attention_size],initializer=tf.glorot_uniform_initializer())
            u = tf.get_variable(name='u',shape=[self.config.word_attention_size],initializer=tf.glorot_uniform_initializer())

            # W = tf.Variable(
            #     tf.truncated_normal([self.config.word_lstm_size*2, self.config.word_attention_size],
            #                         stddev=0.1), name="W")
            # b = tf.Variable(tf.random_normal([self.config.word_attention_size], stddev=0.1),
            #                 name="b")
            # u = tf.Variable(tf.random_normal([self.config.word_attention_size], stddev=0.1),
            #                 name="u")

            att = tf.tanh(
                tf.nn.xw_plus_b(tf.reshape(self.word_outputs, [-1, self.config.word_lstm_size*2]),
                                W, b),
                name="attention_projection"
            )
            logits = tf.matmul(att, tf.reshape(u, [-1, 1]), name="attention_logits")
            attention_weights = tf.nn.softmax(
                tf.reshape(logits, [-1, self.config.word_sequence_length]),
                dim=1,
                name="attention_weights")

            weighted_rnn_output = tf.multiply(
                self.word_outputs, tf.reshape(attention_weights, [-1, self.config.word_sequence_length, 1]),
                name="weighted_rnn_outputs")
            self.word_attention_outputs = tf.reduce_sum(
                weighted_rnn_output, 1, name="word_attention_outputs")
            # self.word_attention_outputs = tf.reduce_max(self.word_outputs, 1)

            self.word_attention_outputs = tf.nn.dropout(
                self.word_attention_outputs, self.config.word_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("feature"):

            fc_mean, fc_var = tf.nn.moments(self.input_feature, axes=[0])
            scale = tf.Variable(tf.ones([1]))
            shift = tf.Variable(tf.zeros([1]))
            epsilon = 0.001

            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)

            mean, var = mean_var_with_update()
            input_feature = tf.nn.batch_normalization(self.input_feature, mean, var, shift, scale, epsilon)

            W = tf.get_variable(name='f_W', shape=[self.config.feature_size , self.config.feature_mlp_size],
                                initializer=tf.glorot_uniform_initializer())
            b = tf.get_variable(name='f_b', shape=[self.config.feature_mlp_size],
                                initializer=tf.glorot_uniform_initializer())

            # W = tf.Variable(
            #     tf.truncated_normal([self.config.feature_size , self.config.feature_mlp_size],
            #                         stddev=0.1), name="W")
            # b = tf.Variable(tf.random_normal([self.config.feature_mlp_size], stddev=0.1),name="b")

            # axis = list(range(2 - 1))
            # wb_mean, wb_var = tf.nn.moments(self.input_feature,axes=[1])
            # scale = tf.Variable(tf.ones([self.config.feature_size]))
            # offset = tf.Variable(tf.zeros([self.config.feature_size]))
            # variance_epsilon = 0.001
            # self.input_feature = tf.nn.batch_normalization(self.input_feature, wb_mean, wb_var, offset, scale, variance_epsilon)

            feature_output=tf.nn.xw_plus_b(input_feature, W, b)
            self.feature_output = tf.nn.dropout(
                feature_output, self.config.feature_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("char_word_feature_MLP"):
            char_word_feature_outputs = tf.concat([self.char_attention_outputs,self.word_attention_outputs,self.feature_output], axis=-1)
            wb_mean, wb_var = tf.nn.moments(char_word_feature_outputs, axes=[0])
            scale = tf.Variable(tf.ones([char_word_feature_outputs.shape[1].value]))
            offset = tf.Variable(tf.zeros([char_word_feature_outputs.shape[1].value]))
            variance_epsilon = 0.001
            char_word_feature_outputs = tf.nn.batch_normalization(char_word_feature_outputs, wb_mean, wb_var, offset, scale,
                                                           variance_epsilon)

            W = tf.get_variable(name='m_W', shape=[char_word_feature_outputs.shape[1].value, self.config.MLP_size],
                                initializer=tf.glorot_uniform_initializer())
            b = tf.get_variable(name='m_b', shape=[self.config.MLP_size],
                                initializer=tf.glorot_uniform_initializer())

            # W = tf.Variable(
            #     tf.truncated_normal(
            #         [char_word_feature_outputs.shape[1].value, self.config.MLP_size], stddev=0.1
            #     ),
            #     name="W"
            # )
            # b = tf.Variable(tf.random_normal([self.config.MLP_size], stddev=0.1), name="b")
            self.mlp_output=tf.nn.xw_plus_b(char_word_feature_outputs, W, b, name="scores")
            # wb_mean, wb_var = tf.nn.moments(self.mlp_output, axes=[0])
            # scale = tf.Variable(tf.ones([self.mlp_output.shape[1].value]))
            # offset = tf.Variable(tf.zeros([self.mlp_output.shape[1].value]))
            # variance_epsilon = 0.001
            # self.mlp_output_BN = tf.nn.batch_normalization(self.mlp_output, wb_mean, wb_var, offset,
            #                                                       scale,
            #                                                       variance_epsilon)

            self.mlp_output=tf.nn.dropout(
                self.mlp_output, self.config.mlp_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("output"):

            W = tf.get_variable(name='o_W', shape=[self.mlp_output.shape[1].value, self.config.num_classes],
                                initializer=tf.glorot_uniform_initializer())
            b = tf.get_variable(name='o_b', shape=[self.config.num_classes],
                                initializer=tf.glorot_uniform_initializer())

            # W = tf.Variable(
            #     tf.truncated_normal(
            #         [self.mlp_output.shape[1].value, self.config.num_classes], stddev=0.1
            #     ),
            #     name="W"
            # )
            # b = tf.Variable(tf.random_normal([self.config.num_classes], stddev=0.1), name="b")
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.mlp_output, W, b, name="scores")

            # self.pre = tf.nn.relu(self.scores)
            # self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # CalculateMean cross-entropy loss
            # with tf.name_scope("loss"):
            #     losses = tf.pow(tf.subtract(self.pre, self.input_label), 2.0)
            #     self.loss = tf.reduce_mean(losses)  # + l1_regularizer(config.l1_reg_lambda, scope='W')(W))
                # self.loss = tf.reduce_mean(losses+l1_regularizer(config.l1_reg_lambda, scope='W')(W)) + self.config.l2_reg_lambda * l2_loss

            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.scores, labels=self.input_label
                )
                self.loss = tf.reduce_mean(losses) #+ self.config.l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_label, 1))
                self.accuracy = tf.reduce_mean(
                    tf.cast(correct_predictions, "float"),
                    name="accuracy")
                recall = tf.cast(tf.reduce_sum(
                    tf.cast(tf.div(tf.add(tf.cast(correct_predictions, dtype=tf.int64), tf.argmax(self.input_label, 1)), 2),
                            dtype=tf.int64)), dtype=tf.float32)
                true_total = tf.cast(tf.reduce_sum(tf.argmax(self.input_label, 1)), dtype=tf.float32)
                self.recall = tf.div(recall, true_total, name='recall')

            # with tf.name_scope("accuracy"):
            #     self.pre_label = tf.cast(self.pre, dtype=tf.int64)
            #     correct_predictions = tf.equal(self.pre_label, tf.cast(self.input_label, dtype=tf.int64))
            #     self.accuracy = tf.reduce_mean(
            #         tf.cast(correct_predictions, "float"),
            #         name="accuracy")
            #     recall = tf.cast(tf.reduce_sum(tf.cast(
            #         tf.div(tf.add(tf.cast(correct_predictions, dtype=tf.int64), tf.cast(self.input_label, dtype=tf.int64)),
            #                2),
            #         dtype=tf.int64)), dtype=tf.float32)
            #     true_total = tf.cast(tf.reduce_sum(self.input_label), dtype=tf.float32)
            #     self.recall = tf.div(recall, true_total, name='recall')
class model6(object):#BiLSTM+Attention+MLP(single) 交叉熵
    def __init__(self,config):
        self.config = config
        self.input_char = tf.placeholder(tf.int32, [None, self.config.char_sequence_length], name="input_char")
        self.input_word  = tf.placeholder(tf.float32, [None, self.config.word_sequence_length,self.config.word_embedding_size], name="input_word")
        self.input_feature=tf.placeholder(tf.float32, [None, self.config.feature_size], name="input_feature")
        self.input_label = tf.placeholder(tf.float32, [None, self.config.num_classes], name="input_label")
        # self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size = tf.placeholder(tf.int32, [], name="batch_size")
        self.real_len = tf.placeholder(tf.int32, [None], name="char_real_len")
        self.real_len_word = tf.placeholder(tf.int32, [None], name="word_real_len")

        with tf.name_scope("char_embedding"):
            _char_embeddings = tf.get_variable(shape=[self.config.char_num,
                                self.config.char_embedding_dim],dtype=tf.float32,name='char_embedding' )
            self.char_embedding = tf.nn.embedding_lookup(_char_embeddings,self.input_char, name="char_embeddings")

        with tf.variable_scope("char_bi-lstm"):
            if self.config.layer_num==1:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                        cell_fw, cell_bw, self.char_embedding,
                        sequence_length=self.real_len, dtype=tf.float32)
                self.char_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # if self.config.dropout:
                #     self.char_outputs = tf.nn.dropout( self.char_outputs, self.config.dropout)
            else:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.char_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                stacked_fw=[]
                stacked_bw = []
                for i in range(self.config.layer_num):
                    stacked_fw.append(cell_fw)
                    stacked_bw.append(cell_bw)
                mcell_fw=tf.nn.rnn_cell.MultiRNNCell(stacked_fw)
                mcell_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_bw)
                (output_fw, output_bw), _ =tf.nn.bidirectional_dynamic_rnn(mcell_fw,mcell_bw,self.char_embedding,
                        sequence_length=self.real_len, dtype=tf.float32)
                self.char_outputs = tf.concat([output_fw, output_bw], axis=-1)

        with tf.name_scope("char_attention"):
            # W = tf.Variable(
            #     tf.truncated_normal([self.config.char_lstm_size*2, self.config.char_attention_size],
            #                         stddev=0.1), name="W")
            # b = tf.Variable(tf.random_normal([self.config.char_attention_size], stddev=0.1),
            #                 name="b")
            # u = tf.Variable(tf.random_normal([self.config.char_attention_size], stddev=0.1),
            #                 name="u")
            #
            # att = tf.tanh(
            #     tf.nn.xw_plus_b(tf.reshape(self.char_outputs, [-1, self.config.char_lstm_size*2]),
            #                     W, b),
            #     name="attention_projection"
            # )
            # logits = tf.matmul(att, tf.reshape(u, [-1, 1]), name="attention_logits")
            # attention_weights = tf.nn.softmax(
            #     tf.reshape(logits, [-1, self.config.char_sequence_length]),
            #     dim=1,
            #     name="attention_weights")
            #
            # weighted_rnn_output = tf.multiply(
            #     self.char_outputs, tf.reshape(attention_weights, [-1, self.config.char_sequence_length, 1]),
            #     name="weighted_rnn_outputs")
            # self.char_attention_outputs = tf.reduce_sum(
            #     weighted_rnn_output, 1, name="attention_outputs")
            self.char_attention_outputs = tf.reduce_max(self.char_outputs, 1)
            self.char_attention_outputs = tf.nn.dropout(
                self.char_attention_outputs, self.config.char_dropout_keep_prob,
                name="dropout")

        with tf.variable_scope("word_bi-lstm"):
            if self.config.layer_num == 1:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, cell_bw, self.input_word,
                    sequence_length=self.real_len, dtype=tf.float32)
                self.word_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # if self.config.dropout:
                #     self.word_outputs = tf.nn.dropout(self.word_outputs, self.config.dropout)
            else:
                cell_fw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                cell_bw = tf.contrib.rnn.LSTMCell(self.config.word_lstm_size)
                if self.config.lstm_dropout_keep_prob:
                    cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                    cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=self.config.lstm_dropout_keep_prob)
                stacked_fw = []
                stacked_bw = []
                for i in range(self.config.layer_num):
                    stacked_fw.append(cell_fw)
                    stacked_bw.append(cell_bw)
                mcell_fw = tf.nn.rnn_cell.MultiRNNCell(stacked_fw)
                mcell_bw = tf.nn.rnn_cell.MultiRNNCell(stacked_bw)
                (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(mcell_fw, mcell_bw, self.input_word,
                                                                            sequence_length=self.real_len_word,
                                                                            dtype=tf.float32)
                self.word_outputs = tf.concat([output_fw, output_bw], axis=-1)
                # self.word_outputs = tf.nn.dropout(self.word_outputs, self.config.dropout)

        with tf.name_scope("word_attention"):
            # Attention mechanism
            W = tf.Variable(
                tf.truncated_normal([self.config.word_lstm_size*2, self.config.word_attention_size],
                                    stddev=0.1), name="W")
            b = tf.Variable(tf.random_normal([self.config.word_attention_size], stddev=0.1),
                            name="b")
            u = tf.Variable(tf.random_normal([self.config.word_attention_size], stddev=0.1),
                            name="u")

            att = tf.tanh(
                tf.nn.xw_plus_b(tf.reshape(self.word_outputs, [-1, self.config.word_lstm_size*2]),
                                W, b),
                name="attention_projection"
            )
            logits = tf.matmul(att, tf.reshape(u, [-1, 1]), name="attention_logits")
            attention_weights = tf.nn.softmax(
                tf.reshape(logits, [-1, self.config.word_sequence_length]),
                dim=1,
                name="attention_weights")

            weighted_rnn_output = tf.multiply(
                self.word_outputs, tf.reshape(attention_weights, [-1, self.config.word_sequence_length, 1]),
                name="weighted_rnn_outputs")
            self.word_attention_outputs = tf.reduce_sum(
                weighted_rnn_output, 1, name="word_attention_outputs")
            # self.word_attention_outputs = tf.reduce_max(self.word_outputs, 1)

            self.word_attention_outputs = tf.nn.dropout(
                self.word_attention_outputs, self.config.word_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("feature"):

            fc_mean, fc_var = tf.nn.moments(self.input_feature, axes=[0])
            scale = tf.Variable(tf.ones([1]))
            shift = tf.Variable(tf.zeros([1]))
            epsilon = 0.001

            ema = tf.train.ExponentialMovingAverage(decay=0.5)

            def mean_var_with_update():
                ema_apply_op = ema.apply([fc_mean, fc_var])
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(fc_mean), tf.identity(fc_var)

            mean, var = mean_var_with_update()
            input_feature = tf.nn.batch_normalization(self.input_feature, mean, var, shift, scale, epsilon)

            W = tf.Variable(
                tf.truncated_normal([self.config.feature_size , self.config.feature_mlp_size],
                                    stddev=0.1), name="W")
            b = tf.Variable(tf.random_normal([self.config.feature_mlp_size], stddev=0.1),name="b")

            # axis = list(range(2 - 1))
            # wb_mean, wb_var = tf.nn.moments(self.input_feature,axes=[1])
            # scale = tf.Variable(tf.ones([self.config.feature_size]))
            # offset = tf.Variable(tf.zeros([self.config.feature_size]))
            # variance_epsilon = 0.001
            # self.input_feature = tf.nn.batch_normalization(self.input_feature, wb_mean, wb_var, offset, scale, variance_epsilon)

            feature_output=tf.nn.xw_plus_b(input_feature, W, b)
            self.feature_output = tf.nn.dropout(
                feature_output, self.config.feature_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("char_word_feature_MLP"):
            char_word_feature_outputs = tf.concat([self.char_attention_outputs,self.word_attention_outputs,self.feature_output], axis=-1)
            wb_mean, wb_var = tf.nn.moments(char_word_feature_outputs, axes=[0])
            scale = tf.Variable(tf.ones([char_word_feature_outputs.shape[1].value]))
            offset = tf.Variable(tf.zeros([char_word_feature_outputs.shape[1].value]))
            variance_epsilon = 0.001
            char_word_feature_outputs = tf.nn.batch_normalization(char_word_feature_outputs, wb_mean, wb_var, offset, scale,
                                                           variance_epsilon)

            W = tf.Variable(
                tf.truncated_normal(
                    [char_word_feature_outputs.shape[1].value, self.config.MLP_size], stddev=0.1
                ),
                name="W"
            )
            b = tf.Variable(tf.random_normal([self.config.MLP_size], stddev=0.1), name="b")
            self.mlp_output=tf.nn.xw_plus_b(char_word_feature_outputs, W, b, name="scores")
            # wb_mean, wb_var = tf.nn.moments(self.mlp_output, axes=[0])
            # scale = tf.Variable(tf.ones([self.mlp_output.shape[1].value]))
            # offset = tf.Variable(tf.zeros([self.mlp_output.shape[1].value]))
            # variance_epsilon = 0.001
            # self.mlp_output_BN = tf.nn.batch_normalization(self.mlp_output, wb_mean, wb_var, offset,
            #                                                       scale,
            #                                                       variance_epsilon)

            self.mlp_output=tf.nn.dropout(
                self.mlp_output, self.config.mlp_dropout_keep_prob,
                name="dropout")

        with tf.name_scope("output"):
            W = tf.Variable(
                tf.truncated_normal(
                    [self.mlp_output.shape[1].value, self.config.num_classes], stddev=0.1
                ),
                name="W"
            )
            b = tf.Variable(tf.random_normal([self.config.num_classes], stddev=0.1), name="b")
            l2_loss = tf.constant(0.0)
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.mlp_output, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            # CalculateMean cross-entropy loss
            with tf.name_scope("loss"):
                losses = tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.scores, labels=self.input_label
                )
                self.loss = tf.reduce_mean(losses) #+ self.config.l2_reg_lambda * l2_loss

            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_label, 1))
                self.accuracy = tf.reduce_mean(
                    tf.cast(correct_predictions, "float"),
                    name="accuracy")
                recall = tf.cast(tf.reduce_sum(
                    tf.cast(tf.div(tf.add(tf.cast(correct_predictions, dtype=tf.int64), tf.argmax(self.input_label, 1)), 2),
                            dtype=tf.int64)), dtype=tf.float32)
                true_total = tf.cast(tf.reduce_sum(tf.argmax(self.input_label, 1)), dtype=tf.float32)
                self.recall = tf.div(recall, true_total, name='recall')