import os
import numpy as np

PC_TRAINING_TESTING =False

#FFT parameters settings
NFFT = 2048
hop_length_FFT = 512
fs = 16e3

label_length = int(np.floor(NFFT/2.+1))*2

N_MFCC = 80

context = 10

#Audio truncation flag and size
audio_reshape = 1
audio_reshape_size = 48000

# if PC_TRAINING_TESTING:
#     current_directory = os.getcwd()
# else:
#     current_directory = os.getcwd()+'/Projects/MFCC_STFT_Framework'

# Here we assume that the directory of 'MFCC_STFT.py' is either the same as for this file or one directory above.
current_directory = os.path.dirname(os.path.abspath(__file__))
if not os.path.isfile(os.path.join(current_directory, 'MFCC_STFT.py')):
    current_directory = os.path.dirname(current_directory)

NETWORK_TYPE = 'AE'

dataset_name = 'Subset'
dataset_type = '_Clean_Speech'

#Tr-Te directory. Output directory
training_dataset_dir = current_directory+'/Audio_Test/Training_'+dataset_name+'/Training_'+dataset_name+dataset_type
testing_dataset_dir = current_directory+'/Audio_Test/Testing_'+dataset_name+'/Testing_'+dataset_name+dataset_type
#testing_dataset_dir=training_dataset_dir
trained_network = current_directory+'/Networks/MFCC_STFT_Network_'+NETWORK_TYPE+'_'+dataset_name+dataset_type
output_network_parameter_file = current_directory+'/Networks/Flue_Pipe_Network_'+NETWORK_TYPE+'_'+dataset_type+'.txt'
output_network_parameter_file_history = current_directory+'/Networks/Flue_Pipe_Network_'+NETWORK_TYPE+'_'+dataset_type+'_HISTORY.txt'

systematic_save_output_folder = current_directory+'/TRAINING_TESTING_HISTORY_SYS_SAVE'

output_recap_file = 'Output_metrics_file.txt'
history_file = 'log_training_MFCC_STFT.txt'

#File format used
audio_format = ".wav"
# features_format = ".p"

#Training FLAGS and SETTINGS
MAKE_DATASET = 1                #If 1 dataset files are made from folders. If 0 dataset are imported from folders.
TRAIN_TEST = 1                  #Training or testing
VALIDATION = 1                  #If 1 we train and validate on a validation split

NORMALIZATION = 1
DENORMALIZATION = 1

LOAD_MODEL = True

#NETWORK AND TRAINING PARAMETERS
optimizer = 'adamax'
activations = 'relu'
final_layer_activation = 'linear'
maxEpochs = 5000
ES_epochs = 500
dropout = 0
learn_rate = 1e-5
MomentumMax = 0.9
decay = 1e-6
minibatch_size = 400
batchnormalization = -1

feature_range_normalization = (-1,1)

CUSTOM_COST_FUNCTION = False


if NETWORK_TYPE == 'MLP':

    layer_dim = [256, 256, 256, label_length]
    N_Layers = len(layer_dim)
    dropout_vec = []


if NETWORK_TYPE == 'AE':

    frame_length = 2048

    audio_reshape_size = frame_length*(audio_reshape_size/frame_length)

    layer_dim = [128, 256, 256, 128]
    N_layers_MLP = len(layer_dim)
    N_filters = [32, 64, 128]
    N_Layers_Conv = len(N_filters)
    CNN_kernel_sizes = [3, 3, 3]
    maxPooling = [2, 2, 2]

    N_Layers = len(layer_dim)

if NETWORK_TYPE == 'CNN':

    layerSizes = [4, 4]
    poolsize = [[-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1], [-1,-1]]
    last_max_pooling_over_time = -1
    receptiveField = [[2,2], [2,2], [2,2], [2,2], [2,2], [2,2], [2,2], [2,2]]

    CNN_layerSize = [4, 4, 8, 8, -1, -1, -1]
    dense_final_layer_size = [64, -1, -1, label_length]
