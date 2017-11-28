
import sys
import os

import cv2
import tensorflow as tf
import numpy as np

from math import floor
from data import get_batch
from config import *

video_path = MODEL_PATH + '/video/'
steps_testing = int(FRAMES_NUM / BATCH_SIZE)
steps_mix = int(MIXED_TOTAL_LENGTH / BATCH_SIZE)

def get_collection_rnn_state(name, ):
    layers = []
    coll = tf.get_collection(name)
    for i in range(0, len(coll), 2):
        state = tf.nn.rnn_cell.LSTMStateTuple(coll[i], coll[i+1])
        layers.append(state)
    return tuple(layers)

def write_video(name, batch):
    print('Writing ' + name + ' video')
    writer = cv2.VideoWriter(video_path + name + '.avi', cv2.VideoWriter_fourcc(*'PIM1'), FRAMERATE,
                             (INPUT_WIDTH, INPUT_HEIGHT), False)
    for i in range(len(batch)):
        writer.write(np.reshape(np.around(batch[i] * 255), (INPUT_HEIGHT, INPUT_WIDTH)).astype('uint8'))
    writer.release()
    print("Done")

def write_mixed_video(sess, x, y_, lstm_init_state, y_net, mse, current_state):
    _current_state = np.zeros((LSTM_INFO[1], 2, BATCH_SIZE, LSTM_INFO[0]))
    accumulator = []
    pointer_v = 0
    print('Creating mixed video data')

    if DATASET_NAME == 'moving_mnist':
        get_batch('reset_test', 'moving_mnist', BATCH_SIZE, None, None, None)
    if DATASET_NAME == 'moving_mnist_sin':
        get_batch('reset_test', 'moving_mnist_sin', BATCH_SIZE, None, None, None)

    for i in range(steps_mix):
        # obtain testing batch
        batch_i, batch_t, pointer_v = get_batch('testing', DATASET_NAME, BATCH_SIZE,
                                                pointer_v, INPUT_WIDTH, INPUT_HEIGHT)
        if pointer_v == 0:
            _current_state = np.zeros((LSTM_INFO[1], 2, BATCH_SIZE, LSTM_INFO[0]))

        # evaluation step
        _y_net, _mse, _current_state = sess.run([y_net, mse, current_state],
                                                feed_dict={x: batch_i, y_: batch_t, lstm_init_state: _current_state})

        if DATASET_NAME == "moving_mnist" and ((i+1) * BATCH_SIZE) % (2 * MIXED_SEQUENCE_LENGTH) == 0:
            get_batch('reset_test', 'moving_mnist', BATCH_SIZE, None, None, None)
        if DATASET_NAME == "moving_mnist_sin" and ((i+1) * BATCH_SIZE) % (2 * MIXED_SEQUENCE_LENGTH) == 0:
            get_batch('reset_test', 'moving_mnist_sin', BATCH_SIZE, None, None, None)

        if accumulator == []:
            accumulator = batch_t
        else:
            if floor(i * BATCH_SIZE / MIXED_SEQUENCE_LENGTH) % 2 == 1:
                accumulator = np.append(accumulator, _y_net, axis=0)
            else:
                accumulator = np.append(accumulator, batch_t, axis=0)
        print('step {} of {}, error: {}'.format(i, steps_mix, _mse))
    write_video('mix', accumulator)

def write_parallel_video(sess, x, y_, lstm_init_state, y_net, mse, current_state):
    _current_state = np.zeros((LSTM_INFO[1], 2, BATCH_SIZE, LSTM_INFO[0]))
    accumulator = []
    pointer_v = 0
    print('Creating parallel video data')

    if DATASET_NAME == 'moving_mnist':
        get_batch('reset_pointer', 'moving_mnist', BATCH_SIZE, None, None, None)
    if DATASET_NAME == 'moving_mnist_sin':
        get_batch('reset_pointer', 'moving_mnist_sin', BATCH_SIZE, None, None, None)

    for i in range(steps_testing):
        # obtain testing batch
        batch_i, batch_t, pointer_v = get_batch('testing', DATASET_NAME, BATCH_SIZE,
                                                pointer_v, INPUT_WIDTH, INPUT_HEIGHT)
        if pointer_v == 0:
            _current_state = np.zeros((LSTM_INFO[1], 2, BATCH_SIZE, LSTM_INFO[0]))

        # evaluation step
        _y_net, _mse, _current_state = sess.run([y_net, mse, current_state],
                                                feed_dict={x: batch_i, y_: batch_t, lstm_init_state: _current_state})

        if accumulator == []:
            accumulator = _y_net
        else:
            accumulator = np.append(accumulator, _y_net, axis=0)
        print('step {} of {}, error: {}'.format(i, steps_testing, _mse))
    write_video('parallel', accumulator)

def get_pattern(n_orig, n_net):
    pointer_n = 0
    mode_orig = True
    check_carry = False
    pattern = []
    carries = []

    for i in range(FRAMES_NUM):
        if check_carry:
            if mode_orig:
                carries = carries + [False]
            else:
                carries = carries + [True]
            check_carry = False

        if (i+1) % BATCH_SIZE == 0:
            check_carry = True

        if mode_orig:
            pattern = pattern + [True]
            pointer_n = pointer_n + 1
            if pointer_n == n_orig:
                pointer_n = 0
                mode_orig = False
        else:
            pattern = pattern + [False]
            pointer_n = pointer_n + 1
            if pointer_n == n_net:
                pointer_n = 0
                mode_orig = True

    return np.reshape(pattern, (steps_testing, BATCH_SIZE)), carries

def write_interlaced_video(sess, n_orig, n_net, x, y_, lstm_init_state, y_net, mse, current_state):
    _current_state = np.zeros((LSTM_INFO[1], 2, BATCH_SIZE, LSTM_INFO[0]))
    accumulator = []
    pointer_v = 0
    last_output_frame = None
    print('Creating interlaced video data')

    if DATASET_NAME == 'moving_mnist':
        get_batch('reset_pointer', 'moving_mnist', BATCH_SIZE, None, None, None)
    if DATASET_NAME == 'moving_mnist_sin':
        get_batch('reset_pointer', 'moving_mnist_sin', BATCH_SIZE, None, None, None)

    pattern, carries = get_pattern(n_orig, n_net)

    for i in range(steps_testing):
        sub_pointer = 0
        sub_pattern = pattern[i]

        # obtain testing batch
        batch_i, batch_t, pointer_v = get_batch('testing', DATASET_NAME, BATCH_SIZE,
                                                  pointer_v, INPUT_WIDTH, INPUT_HEIGHT)
        if pointer_v == 0:
            _current_state = np.zeros((LSTM_INFO[1], 2, BATCH_SIZE, LSTM_INFO[0]))

        if i > 0 and carries[i-1]:
            batch_i[0] = last_output_frame
            sub_pointer = 1

        while sub_pointer < BATCH_SIZE:
            if sub_pattern[sub_pointer]:
                sub_pointer = sub_pointer + 1
                continue

            # evaluation step
            _y_net = y_net.eval(feed_dict={x: batch_i, lstm_init_state: _current_state})

            batch_i[sub_pointer] = _y_net[sub_pointer - 1]
            sub_pointer = sub_pointer + 1

        _y_net, _mse, _current_state = sess.run([y_net, mse, current_state], feed_dict={x: batch_i, y_: batch_t,
                                                                             lstm_init_state: _current_state})
        last_output_frame = _y_net[-1]

        if accumulator == []:
            accumulator = _y_net
        else:
            accumulator = np.append(accumulator, _y_net, axis=0)
        print('step {} of {}, error: {}'.format(i, steps_testing, _mse))

    write_video('interlaced-' + str(n_orig) + '-' + str(n_net), accumulator)

def main(_):
    if not os.path.exists(video_path):
        os.makedirs(video_path)

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(MODEL_PATH + MODEL_NAME + '.meta')
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))

        graph = tf.get_default_graph()
        lstm_init_state = graph.get_tensor_by_name("lstm_state:0")
        x = graph.get_tensor_by_name("input:0")
        y_ = graph.get_tensor_by_name("target:0")
        y_net = tf.get_collection("y_net")[0]
        if ADDITIONAL_OUTPUT:
            mse = tf.get_collection("mse1")[0]
        else:
            mse = tf.get_collection("mse")[0]
        train_step = tf.get_collection("train_step")[0]
        current_state = get_collection_rnn_state("current_state")

        assert FRAMES_NUM % BATCH_SIZE == 0, "Number of frames should be a multiple " \
                                             "of the batch_size used while training"

        if DATASET_NAME == 'moving_mnist':
            get_batch('reset_test', 'moving_mnist', BATCH_SIZE, None, None, None)
        if DATASET_NAME == 'moving_mnist_sin':
            get_batch('reset_test', 'moving_mnist_sin', BATCH_SIZE, None, None, None)
        # obtain original batch
        batch_i, _, _ = get_batch('testing', DATASET_NAME, FRAMES_NUM,
                                  -1, INPUT_WIDTH, INPUT_HEIGHT)

        write_video('original', batch_i)
        write_parallel_video(sess, x, y_, lstm_init_state, y_net, mse, current_state)
        for n in INTERLACED_N:
            write_interlaced_video(sess, n[0], n[1], x, y_, lstm_init_state,
                                   y_net, mse, current_state)
        if MIXED_VIDEO:
            write_mixed_video(sess, x, y_, lstm_init_state, y_net, mse, current_state)

if __name__ == '__main__':
    tf.app.run(main=main, argv=[sys.argv[0]])
