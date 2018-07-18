#!/usr/bin/python
"""
textrnn with word attention
"""
from __future__ import print_function, division
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from models.textrnn.attention import attention
from preprocessing import *
from sklearn.model_selection import *
import pickle
import os
#params
tf.app.flags.DEFINE_string('porn_path','data/porn.txt','pornpath')
tf.app.flags.DEFINE_string('unporn_path','data/unporn.txt','unpornpath')
tf.app.flags.DEFINE_string('modelpath','rnnmodel/','modelpath')
tf.app.flags.DEFINE_float('split_train_test_rate',0.2,'')
tf.app.flags.DEFINE_float('split_train_valid_rate',0.1,'')
tf.app.flags.DEFINE_integer('num_classes',2,'')
tf.app.flags.DEFINE_integer('eval_every_step',1000,'calculate the accuracy of  the valid dataset ')
tf.app.flags.DEFINE_string('log_dir','rnnlogs/','log_dir')

#super params
tf.app.flags.DEFINE_float('keep_prob',0.5,'keep_prob')
tf.app.flags.DEFINE_integer('epochs',30,'epochs')
tf.app.flags.DEFINE_integer('batch_size',256,'batch_size')
tf.app.flags.DEFINE_integer('sequence_length',30,'sequence_length')
tf.app.flags.DEFINE_integer('hidden_size',128,'hidden_size')
tf.app.flags.DEFINE_integer('embedding_size',256,'embedding_size')
tf.app.flags.DEFINE_float('learning_rate',0.001,'learning_rate')
tf.app.flags.DEFINE_integer('attention_size',30,'')
Flags=tf.app.flags.FLAGS



def train(timeprefix):
    # Load the data set
    X, vocab, vocab_to_int, x_shuffled, y_shuffled = load_data(Flags.porn_path, Flags.unporn_path,
                                                               Flags.sequence_length)
    if not os.path.exists(Flags.modelpath):
        os.mkdir(Flags.modelpath)
    with open('vocab'+timeprefix+'.pkl', 'wb')as f:
        pickle.dump(vocab, f)
    with open('vocab_to_int'+timeprefix+'.pkl', 'wb')as f:
        pickle.dump(vocab_to_int, f)
    # 切割数据
    train_X, test_X, train_Y, test_Y = train_test_split(x_shuffled, y_shuffled,
                                                        test_size=Flags.split_train_test_rate, random_state=100)

    train_X, valid_X, train_Y, valid_Y = train_test_split(train_X, train_Y,
                                                          test_size=Flags.split_train_valid_rate, random_state=0)
    print('the shape of the train dataset:{}'.format(train_X.shape))
    print('the shape of the valid dataset:{}'.format(valid_X.shape))
    print('the shape of the test dataset:{}'.format(test_X.shape))

    # Different placeholders
    with tf.name_scope('Inputs'):
        batch_ph = tf.placeholder(tf.int32, [None, Flags.sequence_length], name='batch_ph')
        target_ph = tf.placeholder(tf.float32, [None, Flags.num_classes], name='target_ph')
        seq_len_ph = tf.placeholder(tf.int32, [None], name='seq_len_ph')
        keep_prob_ph = tf.placeholder(tf.float32, name='keep_prob_ph')

    # Embedding layer
    with tf.name_scope('Embedding_layer'):
        embeddings_var = tf.Variable(tf.random_uniform([len(vocab) + 1, Flags.embedding_size], -1.0, 1.0),
                                     trainable=True)
        tf.summary.histogram('embeddings_var', embeddings_var)
        batch_embedded = tf.nn.embedding_lookup(embeddings_var, batch_ph)

    # (Bi-)RNN layer(-s)
    rnn_outputs, _ = bi_rnn(GRUCell(Flags.hidden_size), GRUCell(Flags.hidden_size),
                            inputs=batch_embedded, sequence_length=seq_len_ph, dtype=tf.float32)
    tf.summary.histogram('RNN_outputs', rnn_outputs)

    # Attention layer
    with tf.name_scope('Attention_layer'):
        attention_output, alphas = attention(rnn_outputs, Flags.attention_size, return_alphas=True)
        tf.summary.histogram('alphas', alphas)

    # Dropout
    drop = tf.nn.dropout(attention_output, keep_prob_ph)

    # Fully connected layer
    with tf.name_scope('Fully_connected_layer'):
        W = tf.Variable(tf.truncated_normal([Flags.hidden_size * 2, Flags.num_classes],
                                            stddev=0.1))  # Hidden size is multiplied by 2 for Bi-RNN
        b = tf.Variable(tf.constant(0., shape=[Flags.num_classes]))
        y_hat = tf.nn.xw_plus_b(drop, W, b)
        y_hat = tf.squeeze(y_hat, name='y_hat')
        tf.summary.histogram('W', W)

    with tf.name_scope('Metrics'):
        # Cross-entropy loss and optimizer initialization
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_hat, labels=target_ph))
        tf.summary.scalar('loss', loss)
        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer(learning_rate=Flags.learning_rate).minimize(loss, global_step=global_step)

        # Accuracy metric
        predictions = tf.argmax(y_hat, 1, name='predictions')
        correct_predictions = tf.equal(predictions, tf.argmax(target_ph, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float'), name='accuracy')
        tf.summary.scalar('accuracy', accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(os.path.join(Flags.log_dir, 'train'), accuracy.graph)
    test_writer = tf.summary.FileWriter(os.path.join(Flags.log_dir, 'test'), accuracy.graph)
    session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
    saver = tf.train.Saver()
    with tf.Session(config=session_conf) as sess:
        sess.run(tf.global_variables_initializer())
        print("Start learning...")
        for batch in get_batch_iter(list(zip(train_X, train_Y)), Flags.epochs, Flags.batch_size):
            x_batch, y_batch = zip(*batch)
            # Training
            seq_len = np.array([list(x).index(0) + 1 for x in x_batch])  # actual lengths of sequences
            loss_tr, acc, _, summary = sess.run([loss, accuracy, optimizer, merged],
                                                feed_dict={batch_ph: x_batch,
                                                           target_ph: y_batch,
                                                           seq_len_ph: seq_len,
                                                           keep_prob_ph: Flags.keep_prob})
            current_step = tf.train.global_step(sess, global_step)
            train_writer.add_summary(summary, current_step)
            print('the train dataset:step:{}\tloss:{}\taccuracy:{}'.format(current_step, loss_tr, acc))
            if current_step % Flags.eval_every_step == 0:
                seq_len_valid = np.array([list(x).index(0) + 1 for x in valid_X])  # actual lengths of sequences
                loss_test, accuracy_test = sess.run([loss, accuracy],
                                                    feed_dict={batch_ph: valid_X,
                                                               target_ph: valid_Y,
                                                               seq_len_ph: seq_len_valid,
                                                               keep_prob_ph: 1.})
                print('the valid dataset:the accuracy:{}\tthe loss:{}'.format(accuracy_test, loss_test))
        # cal the accuracy of the test dataset
        seq_len_test = np.array([list(x).index(0) + 1 for x in test_X])  # actual lengths of sequences
        print('the accuracy of the test dataset:{}'.format(sess.run(accuracy,
                                                                    feed_dict={batch_ph: test_X,
                                                                               target_ph: test_Y,
                                                                               seq_len_ph: seq_len_test,
                                                                               keep_prob_ph: 1.})))
        timeprefix_dir=Flags.modelpath+timeprefix+'/'
        if not os.path.exists(timeprefix_dir):  #带有时间戳的目录
            os.mkdir(timeprefix_dir)
        print(timeprefix_dir)
        saver.save(sess, timeprefix_dir+'textrnn.model')


