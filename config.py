
######################
## Model Parameters ##
######################

INPUT_WIDTH = 64

INPUT_HEIGHT = 64

# conv_layer in conv_info : [feature_maps, kernel_size, max_pooling?, max_pooling_size]
CONV_INFO = [[32, 3, False, 2],
             [32, 3, False, 2],
             [64, 3, False, 2]]

# linlayer_info : [use-linlayer?, layer_size]
LINLAYER_INFO = [False, 256]

# lstm_info : [layer_size, num_layers]
LSTM_INFO =  [200, 2]

BATCH_SIZE = 5

# fc_layer in fc_info : layer_size
FC_INFO = [INPUT_WIDTH*INPUT_HEIGHT]

# There is an additional fc layer to serve as the network output
OUTPUT_SIZE = INPUT_WIDTH*INPUT_HEIGHT

# 0: no additional output / 1: prediction + actual frame output / 
# 2: only actual frame output at the end of the net /
# 3: only actual frame output at the end of the conv layers
ADDITIONAL_OUTPUT = 0


#########################
## Training Parameters ##
#########################


MODEL_NAME = 'fish_tank_3'

MODEL_PATH = './models/' + MODEL_NAME + '/'

DATASET_NAME = 'fish_tank'

RETRAINING = False

MIXING = False

MIX_PATH = MODEL_PATH + 'video/mix.avi'

LEARNING_STEP = 1e-4

TRAINING_STEPS = 200000

VALIDATION_STEPS = 100


#########################
## Testing Parameters  ##
#########################

FRAMES_NUM = 200

FRAMERATE = 25

INTERLACED_N = [[1,9], [5,5], [5, 50], [100, 300]]

MIXED_TOTAL_LENGTH = 50000

MIXED_SEQUENCE_LENGTH = 200

MIXED_VIDEO = False

