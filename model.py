
import tensorflow as tf

from math import ceil

def convlstm(x,  input_width, input_height, conv_info, linlayer_info, lstm_init_state,
             lstm_info, batch_size, fc_info, output_size, additional_output):
    h_input = input_layer(x,  input_width, input_height)
    h_conv, downsized_input = conv_layers(h_input, input_width, input_height, conv_info)
    if additional_output == 3:
        y_conv = output_fc_layer(h_conv, downsized_input, fc_info, output_size)
    if linlayer_info[0]:
        h_ll = lin_layer(h_conv, downsized_input, linlayer_info[1])
        h_lstm, lstm_state = lstm_layers(h_ll, lstm_init_state, linlayer_info[1], lstm_info[0], lstm_info[1],
                                         batch_size)
    else:
        h_lstm, lstm_state = lstm_layers(h_conv, lstm_init_state, downsized_input, lstm_info[0], lstm_info[1],
                                         batch_size)
    y_net = output_fc_layer(h_lstm, lstm_info[0], fc_info, output_size)
    y_conv = None
    if additional_output == 1 or additional_output == 2:
        y_conv = output_fc_layer(h_lstm, lstm_info[0], fc_info, output_size)
    return y_net, lstm_state, y_conv

def input_layer(x, input_width, input_height):
    with tf.name_scope('reshape'):
        x_image = tf.reshape(x, [-1, input_width, input_height, 1])
    return x_image

def conv_layers(x_image, input_width, input_height, conv_info):
    # conv_layer in conv_info : [feature maps, kernel size, max-pooling?, max-pooling size]
    h_conv = h_pool = h_out = None
    downsized_width = input_width
    downsized_height = input_height
    for conv_layer, n in zip(conv_info, range(len(conv_info))):
        with tf.name_scope('conv'):
            input_channels = 1 if n == 0 else conv_info[n - 1][0]
            if n == 0:
                input_val = x_image
            elif conv_info[n - 1][2]:
                input_val = h_pool
            else:
                input_val = h_conv

            W_conv = tf.Variable(tf.truncated_normal(
                [conv_layer[1], conv_layer[1], input_channels, conv_layer[0]], stddev=0.1))
            b_conv = tf.Variable(tf.constant(0.1, shape=[conv_layer[0]]))
            h_conv = tf.nn.relu(tf.nn.conv2d(input_val, W_conv, strides=[1, 1, 1, 1], padding='SAME') + b_conv)

            if n == len(conv_info) - 1 and not conv_layer[2]:
                h_out = h_conv
                downsized_input = downsized_width * downsized_height * conv_layer[0]

        if conv_layer[2]:
            with tf.name_scope('maxpool'):
                h_pool = tf.nn.max_pool(h_conv, ksize=[1, conv_layer[3], conv_layer[3], 1],
                                        strides=[1, conv_layer[3], conv_layer[3], 1], padding='SAME')
                downsized_width = ceil(downsized_width / conv_layer[3])
                downsized_height = ceil(downsized_height / conv_layer[3])
                if n == len(conv_info) - 1:
                    downsized_input = downsized_width * downsized_height * conv_layer[0]
                    h_out = h_pool
    return h_out, downsized_input

def lin_layer(h_conv, downsized_input, linlayer_size):
    with tf.name_scope('linlayer'):
        W_ll = tf.Variable(tf.truncated_normal([downsized_input, linlayer_size], stddev=0.1))
        b_ll = tf.Variable(tf.constant(0.1, shape=[linlayer_size]))
        h_pool_flat = tf.reshape(h_conv, [-1, downsized_input])
        h_ll = tf.matmul(h_pool_flat, W_ll) + b_ll
    return h_ll

def lstm_layers(h_ll, init_lstm, linlayer_size, layer_size, num_layers, batch_size):
    with tf.name_scope('lstm'):
        state_per_layer_list = tf.unstack(init_lstm, axis=0)
        rnn_tuple_state = tuple(
             [tf.nn.rnn_cell.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
              for idx in range(num_layers)]
        )

        stacked_lstm = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.BasicLSTMCell(layer_size, state_is_tuple=True)
             for _ in range(num_layers)], state_is_tuple=True)

        h_ll_flat = tf.reshape(h_ll, [batch_size, -1, linlayer_size])
        lstm_out, lstm_state = tf.nn.dynamic_rnn(
            stacked_lstm, h_ll_flat, initial_state=rnn_tuple_state, dtype=tf.float32)

    return lstm_out, lstm_state

def output_fc_layer(h_prev, lstm_size, fc_info, output_size):
    h_prev_size = lstm_size
    for fc_size in fc_info:
        with tf.name_scope('fc'):
            W_fc = tf.Variable(tf.truncated_normal([h_prev_size, fc_size], stddev=0.1))
            b_fc = tf.Variable(tf.constant(0.1, shape=[fc_size]))
            input_flat = tf.reshape(h_prev, [-1, h_prev_size])
            h_prev = tf.matmul(input_flat, W_fc) + b_fc
            h_prev_size = fc_size
    with tf.name_scope('output'):
        W_fc = tf.Variable(tf.truncated_normal([h_prev_size, output_size], stddev=0.1))
        b_fc = tf.Variable(tf.constant(0.1, shape=[output_size]))
        input_flat = tf.reshape(h_prev, [-1, h_prev_size])
        h_out = tf.matmul(input_flat, W_fc) + b_fc
        y_net = tf.nn.sigmoid(h_out)
    return y_net
