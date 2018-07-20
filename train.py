import tensorflow as tf
import numpy as np
import os
import time
import config
import datetime
import logging
from data_utils import *
from rnn_model import *
from tensorflow.contrib import learn
from tensorflow.python.platform import gfile
config=config.Config()
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger = logging.getLogger()
logger.info("Loading data...")
filter_sizes = "3,4,5"
config.filter_sizes=list(map(int, filter_sizes.split(",")))
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        logger.info("Loading rnn model")
        print('Loading rnn model')
        rnn = model1(config)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        if config.optimizer == 'Adam':
            optim = tf.train.AdamOptimizer(learning_rate=config.rate)
        elif config.optimizer == 'Adadelta':
            optim = tf.train.AdadeltaOptimizer(learning_rate=config.rate)
        elif config.optimizer == 'Adagrad':
            optim = tf.train.AdagradOptimizer(learning_rate=config.rate)
        elif config.optimizer == 'RMSProp':
            optim = tf.train.RMSPropOptimizer(learning_rate=config.rate)
        elif config.optimizer == 'Momentum':
            optim = tf.train.MomentumOptimizer(learning_rate=config.rate, momentum=0.9)
        elif config.optimizer == 'SGD':
            optim = tf.train.GradientDescentOptimizer(learning_rate=config.rate)
        else:
            optim = tf.train.GradientDescentOptimizer(learning_rate=config.rate)
        grads_and_vars = optim.compute_gradients(rnn.loss)
        train_op = optim.apply_gradients(grads_and_vars, global_step=global_step)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        if config.checkpoint == "":
            timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            logger.info("Writing to {}\n".format(out_dir))
        else:
            out_dir = config.checkpoint
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        param_dir = os.path.join(os.path.curdir, "runs", timestamp, 'config.py')
        result_dir = os.path.join(os.path.curdir, "runs", timestamp, 'result.txt')
        rnn_dir = os.path.join(os.path.curdir, "runs", timestamp, 'rnn_model.py')
        dev_dir = os.path.join(os.path.curdir, "runs", timestamp, 'dev.txt')
        pf = open(param_dir, 'a', encoding='utf-8')
        rnn_copy= open(rnn_dir , 'a', encoding='utf-8')
        config_file = open('config.py', 'r', encoding='utf-8')
        rnn_file=open('rnn_model.py', 'r', encoding='utf-8')
        config_str = config_file.read()
        rnnstr=rnn_file.read()
        rnn_copy.write(rnnstr)
        rnn_copy.close()
        pf.write(config_str)
        pf.close()
        loss_summary = tf.summary.scalar("loss", rnn.loss)
        acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)
        recall_summary = tf.summary.scalar("recall", rnn.recall)

        # Train Summaries
        train_summary_op = tf.summary.merge([loss_summary, acc_summary, recall_summary,grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

        # Dev summaries

        dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)
        # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=config.num_checkpoints)

        sess.run(tf.global_variables_initializer())

        # ckpt = tf.train.get_checkpoint_state(os.path.join(config.checkpoint, 'checkpoints'))
        # if ckpt and gfile.Exists(ckpt.model_checkpoint_path):
        #     print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        #     saver.restore(sess, ckpt.model_checkpoint_path)
        # saver.restore(sess, './runs/20180522231133/checkpoints/model-28000')
        # saver.restore(sess, './runs/20180601185226/checkpoints/model-4000')
        # saver.restore(sess, './runs/20180618103554/checkpoints/model-14000')
        def train_step(input_char_batch,char_real, input_word_batch,word_real,input_feature_batch,label_batch,config):
            feed_dict = {
                rnn.input_char: input_char_batch,
                rnn.input_word: input_word_batch,
                rnn.batch_size: len(input_char_batch),
                rnn.real_len: char_real,
                rnn.input_feature:input_feature_batch,
                rnn.real_len_word:word_real,
                rnn.input_label:label_batch
            }
            _, step, summaries,loss, accuracy,recall = sess.run(
                [train_op, global_step, train_summary_op, rnn.loss, rnn.accuracy,rnn.recall],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            logger.info(" step {}, loss {:g}, acc {:g},recall {:g}".format(step, loss, accuracy,recall))
            pf = open(result_dir, 'a', encoding='utf-8')
            pf.write("{} {:g} {:g} {:g} \n".format(step, loss, accuracy, recall))
            pf.close()
            train_summary_writer.add_summary(summaries, step)
        def dev_step(config):
            config.is_train=False
            batches = batch_yield( config,df_train,v2id,pos_vec,model,dev_sample_index)
            loss_sum = 0
            accuracy_sum = 0
            count = 0
            recall_sum=0
            config.lstm_dropout_keep_prob = 1.0
            config.word_dropout_keep_prob = 1.0
            config.char_dropout_keep_prob = 1.0
            config.feature_dropout_keep_prob = 1.0
            config.mlp_dropout_keep_prob = 1.0
            for batch in batches:
                char_batch, char_real, word_batch, word_real, feature_batch, label = zip(batch)
                c=char_batch[0]
                w=word_batch[0]
                c_real=char_real[0]
                w_real=word_real[0]
                f_batch=feature_batch[0]
                l=label[0]
                feed_dict = {
                    rnn.input_char: c,
                    rnn.input_word: w,
                    rnn.batch_size: len(c),
                    rnn.real_len: c_real,
                    rnn.input_feature: f_batch,
                    rnn.real_len_word: w_real,
                    rnn.input_label: l
                }
                step,  loss, accuracy,recall = sess.run(
                    [global_step,  rnn.loss, rnn.accuracy,rnn.recall],
                    feed_dict)
                loss_sum = loss_sum + loss
                accuracy_sum = accuracy_sum + accuracy
                recall_sum=recall_sum+recall
                count = count + 1
            loss = loss_sum / count
            accuracy = accuracy_sum / count
            recall=recall_sum/count
            loss_summary = tf.summary.scalar("loss", loss)
            acc_summary = tf.summary.scalar("accuracy", accuracy)
            recall_summary = tf.summary.scalar("recall", recall)
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary, recall_summary])
            summaries_dev=sess.run(dev_summary_op)
            time_str = datetime.datetime.now().isoformat()
            logger.info(" step {}, loss {:g}, acc {:g},recall {:g}".format( step, loss, accuracy, recall))
            pf_dev = open(dev_dir, 'a', encoding='utf-8')
            pf_dev.write("{} {:g} {:g} {:g} \n".format(step, loss, accuracy, recall))
            pf_dev.close()
            dev_summary_writer.add_summary(summaries_dev, step)


        df_train = pd.read_csv(config.trainfile)
        v2id = read_dictionary(config.vocab_path)
        pos_vec = pd.read_csv('data/pos2vec.csv')
        model = word2vec.Word2Vec.load('data/word2vecModel')
        dev_sample_index = -1 * int(config.dev_percentage * float(100000))
        # dev_sample_index = -1
        # dev_step(config)
        batches = batch_yield(config,df_train,v2id,pos_vec,model,dev_sample_index)
        logger.info("With {} batch size, and {} train samples".format(config.batch_size, 90000))
        logger.info("We get {} batches per epoch".format(90000 / config.batch_size))

        # print(batches.shape)
        logger.info("train")
        for batch in batches:
            char_batch, char_real,word_batch,word_real,feature_batch,label = zip(batch)
            # print(char_batch.shape)
            train_step(char_batch[0], char_real[0],word_batch[0],word_real[0],feature_batch[0],label[0],config)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % config.evaluate_every == 0:
                logger.info("\nEvaluation:")
                dev_step(config)
                logger.info("")
            if current_step % config.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                logger.info("Saved model checkpoint to {}\n".format(path))

        logger.info("Final Checkpoint:")
        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
        logger.info("Saved model checkpoint to {}\n".format(path))
                    # path = saver.save(sess, checkpoint_prefix, global_step=current_step)

