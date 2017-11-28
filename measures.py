
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sbn

sbn.set(style="white", palette="muted", color_codes=True)
plt.rcParams['toolbar'] = 'None'

from config import *

video_path = MODEL_PATH + 'video/'

cap = cv2.VideoCapture(video_path + 'original.avi')
#length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
length = FRAMES_NUM
original_video = np.empty((length, INPUT_WIDTH*INPUT_HEIGHT), np.dtype('float32'))
original_range = np.empty((length), np.dtype('float32'))
for i in range(length):
    _, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
    original_video[i] = np.reshape(frame, (-1)) / 255
    original_range[i] = i
cap.release()


differences = np.empty((length-1), np.dtype('float32'))
for i in range(length):
    if i < length - 1:
        differences[i] = ((original_video[i + 1] - original_video[i]) ** 2).mean(axis=None)
plt.plot(original_range[:-1], differences, label='Baseline')
print('Baseline')
print('Max: {}'.format(np.amax(differences)))
print('Min: {}'.format(np.amin(differences)))
print('Mean: {}'.format(np.mean(differences)))
print('StD: {}'.format(np.std(differences)))
print()

for filename in os.listdir(video_path):
    if filename != 'original.avi' and filename != 'mix.avi' and filename[filename.rfind('.'):] == '.avi':
        print(filename)
        cap = cv2.VideoCapture(video_path + filename)
        #length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        length = FRAMES_NUM
        diff2 = 0
        differences = np.empty((length-1), np.dtype('float32'))
        for i in range(length):
            _, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_LINEAR)
            frame = np.reshape(frame, (-1)) / 255
            if i < length - 1:
                differences[i] = ((original_video[i+1] - frame) ** 2).mean(axis=None)
        print('Max: {}'.format(np.amax(differences)))
        print('Min: {}'.format(np.amin(differences)))
        print('Mean: {}'.format(np.mean(differences)))
        print('StD: {}'.format(np.std(differences)))
        print()
        plt.plot(original_range[:-1], differences, label=filename)
        cap.release()
plt.suptitle(MODEL_NAME)
plt.xlabel('frames')
plt.ylabel('MSE')
plt.legend()
#plt.savefig(video_path + 'graph.png')
plt.show()
