
import numpy as np
import cv2

from random import random, randint
from math import floor
from config import MIX_PATH, MIXED_TOTAL_LENGTH

MNIST_LENGTH = 200
MNIST_LENGTH_TEST = 1000

mnist_data = []
last_seq = []
last_point = 0

test_seq = []
test_point = 0

mix_point = 0

def get_batch(mode, dataset, batch_size, pointer, width, height):
    if dataset == 'moving_mnist':
        return moving_mnist(mode, batch_size, False)
    if dataset == 'moving_mnist_sin':
        return moving_mnist(mode, batch_size, True)
    elif dataset == 'random_data':
        return random_data(mode, batch_size, pointer, width, height)
    elif dataset == 'mixed':
        return mixed_data(batch_size, pointer, width, height)
    else:
        if dataset == 'fish_tank':
            return video_data(mode, batch_size, pointer, width, height, './datasets/' + dataset + '/' + dataset + '.mp4',
                              [300, 100000, 105000, 107000])
        else:
            return video_data(mode, batch_size, pointer, width, height, './datasets/' + dataset + '/' + dataset + '.avi',
                              [0, 46000, 47000, 50000])

def mixed_data(batch_size, pointer, width, height):
    cap = cv2.VideoCapture(MIX_PATH)

    cap.set(1, pointer)

    pointer = pointer + batch_size

    batch = np.empty((batch_size + 1, width * height), np.dtype('float32'))
    fc = 0

    while (fc < batch_size + 1):
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        batch[fc] = np.reshape(frame, (-1)) / 255
        fc += 1

    cap.release()

    if pointer + batch_size >= MIXED_TOTAL_LENGTH:
        pointer = 0

    return np.array(batch[:-1]), np.array(batch[1:]), pointer

def lineal_trajectory(length, digit):
    pos_x = randint(0, 36)
    pos_y = randint(0, 36)
    if randint(0, 1) == 0:
        vel_x = 1 + (random()-0.5)/4
    else:
        vel_x = -1 + (random() - 0.5) / 4
    if randint(0, 1) == 0:
        vel_y = 1 + (random()-0.5)/4
    else:
        vel_y = -1 + (random() - 0.5) / 4

    trace = []
    for i in range(length):
        pos_x = pos_x + vel_x
        if floor(pos_x) > (64 - 28):
            pos_x = 64 - 28
            vel_x = -vel_x
        if floor(pos_x) < 0:
            pos_x = 0
            vel_x = -vel_x

        pos_y = pos_y + vel_y
        if floor(pos_y) > (64 - 28):
            pos_y = 64 - 28
            vel_y = -vel_y
        if floor(pos_y) < 0:
            pos_y = 0
            vel_y = -vel_y
        trace = trace + [[pos_x, pos_y]]

    video = []
    for i in range(length):
        frame = np.zeros((64, 64), dtype='uint8')
        pos_x = floor(trace[i][0])
        pos_y = floor(trace[i][1])

        for s in range(28):
            n_pos_y = pos_y + s
            frame[n_pos_y][pos_x:(pos_x+28)] = frame[n_pos_y][pos_x:(pos_x+28)] + digit[s]

        video = video + [frame]

    return video

def sin_trajectory(length, shape):
    shape_size = 28
    phase = randint(0, 179)
    pos_x = randint(0, 64 - shape_size)
    pos_y = randint(0, 64 - shape_size)
    if randint(0, 1) == 0:
        vel_x = 1 - random()
        vel_t_x = 1
    else:
        vel_x = -1 + random()
        vel_t_x = -1
    if randint(0, 1) == 0:
        vel_y = 1 - random()
        vel_t_y = 1
    else:
        vel_y = -1 + random()
        vel_t_y = -1

    trace = []
    for i in range(length):
        pos_x = pos_x + vel_x
        if floor(pos_x) > (64 - shape_size):
            pos_x = 64 - shape_size
            vel_t_x = -vel_t_x
        if floor(pos_x) < 0:
            pos_x = 0
            vel_t_x = -vel_t_x

        pos_y = pos_y + vel_y
        if floor(pos_y) > (64 - shape_size):
            pos_y = 64 - shape_size
            vel_t_y = -vel_t_y
        if floor(pos_y) < 0:
            pos_y = 0
            vel_t_y = -vel_t_y

        vel_x = vel_t_x * abs(np.sin(np.deg2rad(i)))
        vel_y = vel_t_y * abs(np.sin(np.deg2rad(i + phase)))

        trace = trace + [[pos_x, pos_y]]

    video = []
    for i in range(length):
        frame = np.zeros((64, 64), dtype='uint8')
        pos_x = floor(trace[i][0])
        pos_y = floor(trace[i][1])

        for s in range(shape_size):
            n_pos_y = pos_y + s
            frame[n_pos_y][pos_x:(pos_x + shape_size)] = frame[n_pos_y][pos_x:(pos_x + shape_size)] + shape[s]

        video = video + [frame]
    return video

def moving_mnist_dataset(mode, length, sin_move):
    global mnist_data
    if mnist_data == []:
        with open('./datasets/moving_mnist/t10k-images.idx3-ubyte', 'rb') as mnist_file:
            mnist_file.read(16)
            mnist_data = np.fromfile(mnist_file, dtype='uint8')
            mnist_data = np.reshape(mnist_data, (-1, 28, 28))

    if mode == 'testing':
        limits = [9000, 9999]
    else:
        limits = [0, 8999]
            
    if sin_move:
        video1 = sin_trajectory(length, mnist_data[randint(limits[0], limits[1])])
        video2 = sin_trajectory(length, mnist_data[randint(limits[0], limits[1])])
    else:
        video1 = lineal_trajectory(length, mnist_data[randint(limits[0], limits[1])])
        video2 = lineal_trajectory(length, mnist_data[randint(limits[0], limits[1])])
    video = np.clip((np.asarray(video1, dtype='uint') + np.asarray(video2, dtype='uint')),
                    None, 255).astype('uint8')

    return np.reshape(video, (-1, 64*64)) / 255

def moving_mnist(mode, batch_size, sin_move):
    global last_point, last_seq, test_seq, test_point
    if mode == 'training':
        last_point = last_point + batch_size
        if last_seq == [] or last_point == MNIST_LENGTH:
            last_seq = moving_mnist_dataset(mode, MNIST_LENGTH, sin_move)
            last_point = batch_size
        return last_seq[last_point - batch_size:last_point], last_seq[last_point - batch_size + 1:last_point + 1], \
               last_point - batch_size
    elif mode == 'reset_test':
        test_point = 0
        test_seq = moving_mnist_dataset(mode, MNIST_LENGTH_TEST, sin_move)
        return None
    elif mode == 'reset_pointer':
        test_point = 0
        return None
    elif mode == 'testing':
        test_point = test_point + batch_size
        if test_point == MNIST_LENGTH_TEST:
            test_point = batch_size
        return test_seq[test_point - batch_size:test_point], test_seq[test_point - batch_size + 1:test_point + 1], \
               test_point - batch_size
    elif mode == 'validation':
        v_data = moving_mnist_dataset(mode, MNIST_LENGTH, sin_move)
        return v_data[0:batch_size], v_data[1:batch_size + 1], 0
    return None

def video_data(mode, batch_size, pointer, width, height, video_path, sets_limits):
    cap = cv2.VideoCapture(video_path)
    reset = False

    if mode == 'validation':
        pointer = pointer + sets_limits[1]
    elif mode == 'testing':
        pointer = pointer + sets_limits[2]

    if mode == 'training' and pointer+batch_size >= sets_limits[1]:
        reset = True
        pointer = 0
    elif mode == 'validation' and (pointer < sets_limits[1] or pointer+batch_size >= sets_limits[2]):
        reset = True
        pointer = sets_limits[1]
    elif mode == 'testing' and (pointer < sets_limits[2] or pointer+batch_size >= sets_limits[3]):
        reset = True
        pointer = sets_limits[2]

    cap.set(1, pointer)

    batch = np.empty((batch_size + 1, width*height), np.dtype('float32'))
    fc = 0

    while (fc < batch_size + 1):
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
        batch[fc] = np.reshape(frame, (-1)) / 255
        fc += 1

    cap.release()

    if mode == 'validation':
        pointer = pointer - sets_limits[1]
    elif mode == 'testing':
        pointer = pointer - sets_limits[2]

    if not reset:
        pointer = pointer + batch_size

    return np.array(batch[:-1]), np.array(batch[1:]), pointer

def random_data(mode, batch_size, pointer, width, height):
    batch = []
    for n in range(batch_size):
        line = []
        for x in range(width*height):
            line = line + [random()]
        batch = batch + [line]
    return np.array(batch[:-1]), np.array(batch[1:]), 0
