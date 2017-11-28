
import sys
import shutil
import os

import tensorflow as tf
import numpy as np

from model import convlstm
from data import get_batch
from config import *

def get_collection_rnn_state(name):
    layers = []
    coll = tf.get_collection(name)
    for i in range(0, len(coll), 2):
        state = tf.nn.rnn_cell.LSTMStateTuple(coll[i], coll[i+1])
        layers.append(state)
    return tuple(layers)

def main(_):
    zero_state = np.zeros((LSTM_INFO[1], 2, BATCH_SIZE, LSTM_INFO[0]))
    _current_state = _current_state_v = None

    if not RETRAINING:
        # init_state : tf.placeholder(tf.float32, [lstm_num_layers, input_image_dimensions, batch_size, lstm_size])
        lstm_init_state = tf.placeholder(tf.float32, [LSTM_INFO[1], 2, BATCH_SIZE, LSTM_INFO[0]], name="lstm_state")

        # input placeholder
        x = tf.placeholder(tf.float32, [None, (INPUT_WIDTH*INPUT_HEIGHT)], name="input")

        # target placeholder
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name="target")
        if ADDITIONAL_OUTPUT == 1:
            y_c_ = tf.placeholder(tf.float32, [None, OUTPUT_SIZE], name="target-c")

        # building the model
        y_net, current_state, y_conv = convlstm(x=x,  input_width=INPUT_WIDTH, input_height=INPUT_HEIGHT,
                                                conv_info=CONV_INFO, linlayer_info=LINLAYER_INFO,
                                                lstm_init_state=lstm_init_state, lstm_info=LSTM_INFO,
                                                batch_size=BATCH_SIZE, fc_info=FC_INFO, output_size=OUTPUT_SIZE,
                                                additional_output=ADDITIONAL_OUTPUT)
        tf.add_to_collection('y_net', y_net)

        if ADDITIONAL_OUTPUT > 0:
            tf.add_to_collection('y_conv', y_conv)

        for layer in current_state:
            tf.add_to_collection('current_state', layer.c)
            tf.add_to_collection('current_state', layer.h)

        # network loss
        with tf.name_scope('network_loss'):
            if ADDITIONAL_OUTPUT == 1:
                mse1 = tf.losses.mean_squared_error(labels=y_, predictions=y_net)
                mse2 = tf.losses.mean_squared_error(labels=y_c_, predictions=y_conv)
                mse = mse1 + mse2
                tf.add_to_collection('mse1', mse1)
                tf.add_to_collection('mse2', mse2)
            elif ADDITIONAL_OUTPUT == 2 or ADDITIONAL_OUTPUT == 3:
                mse = tf.losses.mean_squared_error(labels=y_, predictions=y_conv)
            else:
                mse = tf.losses.mean_squared_error(labels=y_, predictions=y_net)
            tf.add_to_collection('mse', mse)

        # optimizer
        with tf.name_scope('adam_optimizer'):
            train_step = tf.train.AdamOptimizer(LEARNING_STEP).minimize(mse)
            tf.add_to_collection('train_step', train_step)

        # graph saver
        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        train_writer = tf.summary.FileWriter(MODEL_PATH)
        train_writer.add_graph(tf.get_default_graph())
        print('Saving graph to: ' + MODEL_PATH)

        # saving config file
        shutil.copy('./config.py', MODEL_PATH + '/config.py')

    # training
    with tf.Session() as sess:
        if RETRAINING:
            loader = tf.train.import_meta_graph(MODEL_PATH + MODEL_NAME + '.meta')
            loader.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))

            graph = tf.get_default_graph()
            lstm_init_state = graph.get_tensor_by_name("lstm_state:0")
            x = graph.get_tensor_by_name("input:0")
            y_ = graph.get_tensor_by_name("target:0")
            y_net = tf.get_collection("y_net")[0]

            train_step = tf.get_collection("train_step")[0]
            current_state = get_collection_rnn_state("current_state")

            if ADDITIONAL_OUTPUT == 1:
                y_c_ = graph.get_tensor_by_name("target-c:0")
                y_conv = tf.get_collection("y_conv")[0]
                mse1 = tf.get_collection("mse1")[0]
                mse2 = tf.get_collection("mse2")[0]
            elif ADDITIONAL_OUTPUT == 2 or ADDITIONAL_OUTPUT == 3:
                y_conv = tf.get_collection("y_conv")[0]
            mse = tf.get_collection("mse")[0]
        else:
            sess.run(tf.global_variables_initializer())

        # training saver
        saver = tf.train.Saver()

        _mse = _mse1 = _mse2 = None
        training_p = 0
        validation_v = 0

        for i in range(TRAINING_STEPS):
            # set lstm state to 0 if it is the first step in the sequence
            if training_p == 0:
                _current_state = np.zeros((LSTM_INFO[1], 2, BATCH_SIZE, LSTM_INFO[0]))
            if validation_v == 0:
                _current_state_v = np.zeros((LSTM_INFO[1], 2, BATCH_SIZE, LSTM_INFO[0]))

            # obtain training batch
            if not MIXING:
                batch_i, batch_t, training_p = get_batch('training', DATASET_NAME, BATCH_SIZE,
                                                         training_p, INPUT_WIDTH, INPUT_HEIGHT)
            else:
                batch_i, batch_t, training_p = get_batch('training', 'mixed', BATCH_SIZE,
                                                         training_p, INPUT_WIDTH, INPUT_HEIGHT)

            # validation step
            if i % VALIDATION_STEPS == 0:
                # obtain validation batch
                if not MIXING:
                    batch_v_i, batch_v_t, validation_v = get_batch('validation', DATASET_NAME, BATCH_SIZE,
                                                                   validation_v, INPUT_WIDTH, INPUT_HEIGHT)
                else:
                    batch_v_i, batch_v_t, validation_v = get_batch('validation', 'mixed', BATCH_SIZE,
                                                                   validation_v, INPUT_WIDTH, INPUT_HEIGHT)

                if ADDITIONAL_OUTPUT == 2 or ADDITIONAL_OUTPUT == 3:
                    batch_t = batch_i
                    batch_v_t = batch_v_i

                if ADDITIONAL_OUTPUT == 1:
                    validation_loss, _current_state_v = sess.run([mse, current_state], feed_dict={
                        x: batch_v_i,
                        y_: batch_v_t,
                        y_c_: batch_v_i,
                        lstm_init_state: _current_state_v})

                    print('step {} of {}, training loss1: {} '
                          '// loss2: {} // validation loss: {}'.format(i, TRAINING_STEPS, _mse1, _mse2,
                                                                       validation_loss))
                else:
                    validation_loss, _current_state_v = sess.run([mse, current_state], feed_dict={
                        x: batch_v_i,
                        y_: batch_v_t,
                        lstm_init_state: _current_state_v})

                    print('step {} of {}, training loss: {} // validation loss: {}'.format(i, TRAINING_STEPS, _mse,
                                                                                           validation_loss))

            if ADDITIONAL_OUTPUT == 1:
                # training step
                _mse1, _mse2, _train_step, _current_state = sess.run([mse1, mse2, train_step, current_state],
                                                             feed_dict={x: batch_i, y_: batch_t, y_c_: batch_i,
                                                                        lstm_init_state: _current_state})
            else:
                # training step
                _mse, _train_step, _current_state = sess.run([mse, train_step, current_state],
                    feed_dict={x: batch_i, y_: batch_t, lstm_init_state: _current_state})

            if MIXING and ((i + 1) * BATCH_SIZE) % (2 * MIXED_SEQUENCE_LENGTH) == 0:
                _current_state = np.zeros((LSTM_INFO[1], 2, BATCH_SIZE, LSTM_INFO[0]))
                _current_state_v = np.zeros((LSTM_INFO[1], 2, BATCH_SIZE, LSTM_INFO[0]))

        saver.save(sess, MODEL_PATH + MODEL_NAME)

if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
