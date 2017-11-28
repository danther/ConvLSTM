
import numpy as np
import cv2
import os

from math import floor
from random import random, randint

VIDEO_LENGTH = 50000

def trajectory_to_video(length, trace, shape, shape_size, rotation):
    video = []
    for i in range(length):
        frame = np.zeros((64, 64), dtype='uint8')
        pos_x = floor(trace[i][0])
        pos_y = floor(trace[i][1])

        for s in range(shape_size):
            n_pos_y = pos_y + s
            frame[n_pos_y][pos_x:(pos_x + shape_size)] = frame[n_pos_y][pos_x:(pos_x + shape_size)] + shape[s]
        if rotation:
            shape = np.rot90(shape)

        video = video + [frame]
    return video

def linear_trajectory(length, shape_size):
    pos_x = randint(0, 64 - shape_size)
    pos_y = randint(0, 64 - shape_size)
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
        if floor(pos_x) > (64 - shape_size):
            pos_x = 64 - shape_size
            vel_x = -vel_x
        if floor(pos_x) < 0:
            pos_x = 0
            vel_x = -vel_x

        pos_y = pos_y + vel_y
        if floor(pos_y) > (64 - shape_size):
            pos_y = 64 - shape_size
            vel_y = -vel_y
        if floor(pos_y) < 0:
            pos_y = 0
            vel_y = -vel_y
        trace = trace + [[pos_x, pos_y]]

    return trace

def linear_trajectory_d1(length, shape_size, pos_y):
    pos_x = randint(0, 64 - shape_size)
    if randint(0, 1) == 0:
        vel_x = 1 + (random()-0.5)/4
    else:
        vel_x = -1 + (random() - 0.5) / 4

    trace = []
    for i in range(length):
        pos_x = pos_x + vel_x
        if floor(pos_x) > (64 - shape_size):
            pos_x = 64 - shape_size
            vel_x = -vel_x
        if floor(pos_x) < 0:
            pos_x = 0
            vel_x = -vel_x

        trace = trace + [[pos_x, pos_y]]

    return trace

def sin_trajectory(length, shape_size):
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

    return trace

def get_shape(shape_name, shape_size, invert, low_trim):
    img = cv2.imread('./shapes/' + shape_name, 0)
    img = cv2.resize(img, (shape_size, shape_size), interpolation=cv2.INTER_LINEAR)
    if low_trim > 0:
        img[img < 200] = 0
    img = np.clip(np.asarray(img, dtype='uint') * 255, None, 255).astype('uint8')
    if invert:
        img = np.invert(img, dtype='uint8')
    return img

def write_video(name, video):
    print('Writing ' + name + ' video')
    if not os.path.exists('./datasets/' + name + '/'):
        os.makedirs('./datasets/' + name + '/')
    writer = cv2.VideoWriter('./datasets/' + name + '/' + name + '.avi', cv2.VideoWriter_fourcc(*'PIM1'), 25,
                             (64, 64), False)
    for i in range(len(video)):
        writer.write(np.reshape(np.around(video[i] * 255), (64, 64)).astype('uint8'))
    writer.release()
    print("Done")

def create_video(length, shape_name, shape_size, shape_file, invert=False, rotation=False, low_trim=0, sin_move=False, d1_mov=False, d1_pos=None,
                double_shape=False, double_file=None, double_size=None, double_invert=False, double_low_trim=0):
    # Create shape
    shape = get_shape(shape_file, shape_size, invert, low_trim)

    # Create trajectory
    if d1_mov:
        trajectory = linear_trajectory_d1(length, shape_size, d1_pos)
    elif sin_move:
        trajectory = sin_trajectory(length, shape_size)
    else:
        trajectory = linear_trajectory(length, shape_size)

    # Create raw video
    raw_video = trajectory_to_video(length, trajectory, shape, shape_size, rotation)
    
    if double_shape:
        # Two shapes video
        shape = get_shape(double_file, double_size, double_invert, double_low_trim)
        trajectory = linear_trajectory(length, double_size)
        raw_video2 = trajectory_to_video(length, trajectory, shape, double_size, False)
        video = np.clip((np.asarray(raw_video, dtype='uint') + np.asarray(raw_video2, dtype='uint')),
                    None, 255).astype('uint8')
        video = np.reshape(video, (-1, 64*64)) / 255
    else:
        # One shape video
        video = np.reshape(np.asarray(raw_video, dtype='uint8'), (-1, 64 * 64)) / 255

    write_video(shape_name, video)

create_video(VIDEO_LENGTH, 'circle', 32, 'circle.png')
create_video(VIDEO_LENGTH, 'star', 32, 'star.jpg', invert=True)
create_video(VIDEO_LENGTH, 'triangle', 32, 'triangle.jpg', invert=True, low_trim=200)

create_video(VIDEO_LENGTH, 'star-rot', 32, 'star.jpg', invert=True, rotation=True)
create_video(VIDEO_LENGTH, 'triangle-rot', 32, 'triangle.jpg', invert=True, rotation=True, low_trim=200)

create_video(VIDEO_LENGTH, 'triangle-sin', 32, 'triangle.jpg', invert=True, low_trim=200, sin_move=True)
create_video(VIDEO_LENGTH, 'triangle-rot-sin', 32, 'triangle.jpg', invert=True, rotation=True, low_trim=200, sin_move=True)

create_video(VIDEO_LENGTH, 'circle-1d1', 32, 'circle.png', d1_mov=True, d1_pos=0)
create_video(VIDEO_LENGTH, 'circle-1d2', 32, 'circle.png', d1_mov=True, d1_pos=32)

create_video(VIDEO_LENGTH, 'double_triangle', 32, 'triangle.jpg', invert=True, low_trim=200, double_shape=True, 
             double_file='triangle.jpg', double_size=32, double_invert=True, double_low_trim=200)

create_video(VIDEO_LENGTH, 'triangle-star', 32, 'triangle.jpg', invert=True, low_trim=200, double_shape=True, 
             double_file='star.jpg', double_size=32, double_invert=True)
