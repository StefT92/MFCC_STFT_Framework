"""
MFCC to STFT Neural approach to inverse transform

Requires librosa

Author: S.Tomassetti <tomassetti.ste@gmail.com>, 2017
"""
import numpy as np
import librosa
import os
import wave
import cPickle as pickle
import NetCode.NetCreate as NetCreate
import NetCode.PostUtil as PostUtil
import time
from sklearn import preprocessing
import scipy.io.wavfile as WAV
from keras import backend as K
import NetCode.paper_spectr_plot as paper_spectr_plot
import keras
import theano
import argparse
import imp
from PATH_VAR_INIT import PC_TRAINING_TESTING
import matplotlib.pyplot as plt

print '\nScript Starts on: '+time.strftime("%d-%m-%Y")
print 'Time: '+time.strftime("%H_%M_%S")

print '\nKeras Version = '+keras.__version__

print 'Theano Version = '+theano.__version__

parser = argparse.ArgumentParser()
parser.add_argument("--PATH_VAR_INIT_ref", default="PATH_VAR_INIT.py")
args = parser.parse_args()
PATH_VAR_INIT_ref = args.PATH_VAR_INIT_ref

print '\nConfiguration File Used: '+PATH_VAR_INIT_ref
if PATH_VAR_INIT_ref != "PATH_VAR_INIT.py":
    imp.load_source("PATH_VAR_INIT", PATH_VAR_INIT_ref)


if PC_TRAINING_TESTING:
    plt.switch_backend('Agg')
    # from IPython import embed
else:
    # import matplotlib
    # matplotlib.use('Agg')
    from IPython import embed
    pass


def invlogamplitude(S):
    """librosa.logamplitude is actually 10_log10, so invert that."""
    return 10.0**(S/10.0)


def reconstrMFCC(MFCC, NFFT, hop_length_FFT, fs, n_mfcc, n_mel=128, Pxx_c=None, PLOT=False):

    dctm = librosa.filters.dct(n_mfcc, n_mel)
    mel_basis = librosa.filters.mel(fs, NFFT)
    bin_scaling = 1.0 / np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis),axis=0))
    recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T,invlogamplitude(np.dot(dctm.T, MFCC)))

    recon = (librosa.istft(stft_matrix=recon_stft,win_length=NFFT, hop_length=hop_length_FFT,center=True))
    recon_target = (librosa.istft(stft_matrix=np.abs(Pxx_c),win_length=NFFT, hop_length=hop_length_FFT,center=True))

    # scaled = np.int16(paper_spectr_plot.normalize(recon) / np.max(np.abs(paper_spectr_plot.normalize(recon))) * 32767)
    # WAV.write('/home/stefano/Scrivania/ciao.wav', fs, scaled)

    if PLOT:
        if Pxx_c is None:
            plt.imshow(np.abs(recon_stft),aspect='auto',cmap='gray_r')
            plt.title('MFCC reconstructed STFT')
            plt.show()
        else:
            plt.subplot(221)
            plt.imshow(np.abs(Pxx_c), aspect='auto', cmap='gray_r')
            plt.title('Audio STFT')
            plt.subplot(223)
            plt.plot(recon_target, 'k')
            plt.grid(True)
            plt.title('Target reconstructed waveform')
            plt.subplot(222)
            plt.imshow(np.abs(recon_stft), aspect='auto',cmap='gray_r')
            plt.title('MFCC reconstructed STFT')
            plt.subplot(224)
            plt.plot(recon, 'k')
            plt.grid(True)
            plt.title('MFCC reconstructed waveform')
            plt.show()

    return recon_stft, recon


def custom_cost_function(y_true, y_pred):

    from PATH_VAR_INIT import NFFT

    spec_mat = (y_pred[:,0:NFFT])#+1j*y_pred[:,NFFT:2*NFFT]).astype('float32')
    spec_mat_true = (y_true[:,0:NFFT]) #+1j*y_true[:,NFFT:2*NFFT]).astype('float32')

    lsd=K.mean(K.sqrt(K.mean(K.log(K.abs(spec_mat)/K.abs(spec_mat_true))**2,axis=-1)),axis=-1)
    mse_loss = K.mean(K.square(y_pred - y_true), axis=-1)

    sum_partial_losses = mse_loss+lsd

    return sum_partial_losses


def dataExtraction_MFCC_Recon(NFFT, hop_length_FFT, fs, N_MFCC, training_dataset_dir, testing_dataset_dir, audio_format,audio_reshape, audio_reshape_size, MAKE_DATASET, TRAIN_TEST):

    if MAKE_DATASET == 1:
        print 'Making dataset...'
        file_audio_index = 1
        filename_list_training = []
        lst = os.listdir(training_dataset_dir)
        lst.sort()
        if TRAIN_TEST == 1:
            for filename in lst:
                if filename.endswith(audio_format):
                    filename_list_training.append(filename)
                    spf = wave.open("%s/%s" % (training_dataset_dir, filename))
                    # print filename
                    audio_in = spf.readframes(-1)
                    audio_in = np.fromstring(audio_in, 'Int16')
                    audio_in = audio_in / 2 ** 15.

                    if len(audio_in) >= audio_reshape_size:
                        if audio_reshape == 1:
                            audio_in = audio_in[0:audio_reshape_size]

                        Pxx_c = librosa.stft(audio_in, n_fft=NFFT, win_length=NFFT, hop_length=hop_length_FFT,
                                             center=True, window=np.blackman(NFFT))

                        S = librosa.feature.melspectrogram(S=Pxx_c, sr=fs, n_mels=128, n_fft=NFFT,
                                                           hop_length=hop_length_FFT)
                        MFCCxx = librosa.feature.mfcc(S=librosa.core.logamplitude(S), n_mfcc=N_MFCC, sr=fs)

                        Rxx, audio_r = reconstrMFCC(MFCCxx, NFFT=NFFT, hop_length_FFT=hop_length_FFT, fs=fs,n_mfcc=N_MFCC, n_mel=128, Pxx_c=Pxx_c, PLOT=False)
                        #
                        # plt.subplot(221)
                        # plt.plot(audio_in,'k')
                        # plt.grid(True)
                        # plt.subplot(223)
                        # plt.plot(audio_r,'k')
                        # plt.grid(True)
                        # plt.subplot(222)
                        # plt.imshow(np.abs(Pxx_c), aspect='auto', cmap='gray_r')
                        # plt.subplot(224)
                        # plt.imshow(np.abs(Rxx),aspect='auto',cmap='gray_r')
                        #
                        # scaled = np.int16(paper_spectr_plot.normalize(audio_in) / np.max(np.abs(paper_spectr_plot.normalize(audio_in))) * 32767)
                        # WAV.write('/home/stefano/Scrivania/ciao_in.wav', fs, scaled)
                        #
                        # scaled = np.int16(paper_spectr_plot.normalize(audio_r) / np.max(np.abs(paper_spectr_plot.normalize(audio_r))) * 32767)
                        # WAV.write('/home/stefano/Scrivania/ciao_r.wav', fs, scaled)
                        # plt.show()


                        if file_audio_index == 1:
                            Y_train = np.reshape(audio_in[0:len(audio_r)],[1,-1])
                            X_train = np.reshape(audio_r,[1,-1])
                        else:

                            Y_train = np.concatenate((Y_train, np.reshape(audio_in[0:len(audio_r)],[1,-1])), axis=0)

                            X_train = np.append(X_train, np.reshape(audio_r,[1,-1]), axis=0)

                        file_audio_index = file_audio_index + 1

                    else:
                        continue

                    continue

        filename_list_testing = []
        file_audio_index = 1
        lst = os.listdir(testing_dataset_dir)
        lst.sort()

        for filename in lst:
            if filename.endswith(audio_format):
                filename_list_testing.append(filename)
                spf = wave.open("%s/%s" % (testing_dataset_dir, filename))
                # print filename
                audio_in = spf.readframes(-1)
                audio_in = np.fromstring(audio_in, 'Int16')
                audio_in = audio_in / 2 ** 15.
                if len(audio_in) >= audio_reshape_size:

                    if audio_reshape == 1:
                        audio_in = audio_in[0:audio_reshape_size]

                    Pxx_c = librosa.stft(audio_in, n_fft=NFFT, win_length=NFFT, hop_length=hop_length_FFT,
                                         center=True, window=np.blackman(NFFT))

                    S = librosa.feature.melspectrogram(S=Pxx_c, sr=fs, n_mels=128, n_fft=NFFT,
                                                       hop_length=hop_length_FFT)
                    MFCCxx = librosa.feature.mfcc(S=librosa.core.logamplitude(S), n_mfcc=N_MFCC, sr=fs)

                    Rxx, audio_r = reconstrMFCC(MFCCxx, NFFT=NFFT, hop_length_FFT=hop_length_FFT, fs=fs, n_mfcc=N_MFCC,
                                                n_mel=128, Pxx_c=Pxx_c, PLOT=False)

                    if file_audio_index == 1:
                        Y_test = np.reshape(audio_in[0:len(audio_r)],[1,-1])
                        X_test = np.reshape(audio_r,[1,-1])
                    else:
                        Y_test = np.concatenate((Y_test, np.reshape(audio_in[0:len(audio_r)],[1,-1])), axis=0)

                        X_test = np.append(X_test, np.reshape(audio_r,[1,-1]), axis=0)

                    file_audio_index = file_audio_index + 1
                else:
                    continue
                continue

        if TRAIN_TEST == 1:
            pickle.dump(X_train, open("%s/%s" % (training_dataset_dir, 'TRAINING_DATASET.p'), 'wb'))
            pickle.dump(Y_train, open("%s/%s" % (training_dataset_dir, 'TRAINING_LABELS.p'), 'wb'))
            np.savetxt(training_dataset_dir + '/filenames.txt', filename_list_training, fmt='%s', newline='\n')

        elif TRAIN_TEST == 0:
            Y_train = pickle.load(open("%s/%s" % (training_dataset_dir, 'TRAINING_LABELS.p'), 'rb'))
            X_train = np.zeros(1)

        pickle.dump(X_test, open("%s/%s" % (testing_dataset_dir, 'TESTING_DATASET.p'), 'wb'))
        pickle.dump(Y_test, open("%s/%s" % (testing_dataset_dir, 'TESTING_LABELS.p'), 'wb'))
        np.savetxt(testing_dataset_dir + '/filenames.txt', filename_list_testing, fmt='%s', newline='\n')

    else:
        print 'Loading dataset...'
        if TRAIN_TEST == 1:
            X_train = pickle.load(open("%s/%s" % (training_dataset_dir, 'TRAINING_DATASET.p'), 'rb'))
            X_test = pickle.load(open("%s/%s" % (testing_dataset_dir, 'TESTING_DATASET.p'), 'rb'))

            Y_train = pickle.load(open("%s/%s" % (training_dataset_dir, 'TRAINING_LABELS.p'), 'rb'))
            Y_test = pickle.load(open("%s/%s" % (testing_dataset_dir, 'TESTING_LABELS.p'), 'rb'))

        elif TRAIN_TEST == 0:
            X_train = np.zeros(1)
            Y_train = pickle.load(open("%s/%s" % (training_dataset_dir, 'TRAINING_LABELS.p'), 'rb'))

            X_test = pickle.load(open("%s/%s" % (testing_dataset_dir, 'TESTING_DATASET.p'), 'rb'))
            Y_test = pickle.load(open("%s/%s" % (testing_dataset_dir, 'TESTING_LABELS.p'), 'rb'))

    print '\nDataset Extracted:'
    print 'X_train shape', X_train.shape
    print 'Y_train shape', Y_train.shape

    print 'X_test shape', X_test.shape
    print 'Y_test shape', Y_test.shape

    return X_train, Y_train, X_test, Y_test


def dataExtraction(NFFT, hop_length_FFT, fs, N_MFCC, training_dataset_dir, testing_dataset_dir, audio_format, mfcc_shape_file_name, stft_shape_file_name,audio_reshape, audio_reshape_size,MAKE_DATASET, TRAIN_TEST):

    if MAKE_DATASET == 1:
        print 'Making dataset...'
        file_audio_index = 1
        filename_list_training = []
        lst = os.listdir(training_dataset_dir)
        lst.sort()
        if TRAIN_TEST == 1:
            for filename in lst:

                if filename.endswith(audio_format):
                    filename_list_training.append(filename)
                    spf = wave.open("%s/%s" % (training_dataset_dir, filename))
                    # print filename
                    audio_in = spf.readframes(-1)
                    audio_in = np.fromstring(audio_in, 'Int16')
                    audio_in = audio_in / 2 ** 15.

                    if len(audio_in)>=audio_reshape_size:
                        if audio_reshape == 1:
                            audio_in=audio_in[0:audio_reshape_size]

                        Pxx_c = librosa.stft(audio_in, n_fft=NFFT, win_length=NFFT, hop_length=hop_length_FFT, center=True,window=np.blackman(NFFT))

                        Pxx=np.concatenate([np.real(Pxx_c),np.imag(Pxx_c)],axis=0)

                        if file_audio_index == 1:
                            spec_width = Pxx.shape[0]
                            spec_length = Pxx.shape[1]
                            pickle.dump([spec_width, spec_length], open(stft_shape_file_name, 'wb'))
                            tensor_shape = (spec_width, spec_length)
                            Y_train = np.reshape(Pxx, tensor_shape)
                        else:
                            Y_train = np.concatenate((Y_train, np.reshape(Pxx, tensor_shape)),axis=1)

                        S = librosa.feature.melspectrogram(S=Pxx_c, sr=fs, n_mels=128, n_fft=NFFT,hop_length=hop_length_FFT)
                        MFCCxx=librosa.feature.mfcc(S=librosa.core.logamplitude(S),n_mfcc=N_MFCC,sr=fs)

                        if file_audio_index == 1:
                            mfcc_shape = (MFCCxx.shape[0], MFCCxx.shape[1])
                            pickle.dump([MFCCxx.shape[0], MFCCxx.shape[1]], open(mfcc_shape_file_name, 'wb'))
                            X_train = np.reshape(MFCCxx, mfcc_shape)
                        else:
                            X_train = np.append(X_train, np.reshape(MFCCxx, mfcc_shape), axis=1)
                        file_audio_index = file_audio_index + 1

                    else:
                        continue

                    continue

        filename_list_testing =[]
        file_audio_index = 1
        lst = os.listdir(testing_dataset_dir)
        lst.sort()

        for filename in lst:
            if filename.endswith(audio_format):
                filename_list_testing.append(filename)
                spf = wave.open("%s/%s" % (testing_dataset_dir, filename))
                # print filename
                audio_in = spf.readframes(-1)
                audio_in = np.fromstring(audio_in, 'Int16')
                audio_in = audio_in / 2 ** 15.
                if len(audio_in) >= audio_reshape_size:

                    if audio_reshape == 1:
                        audio_in = audio_in[0:audio_reshape_size]

                    Pxx_c = librosa.stft(audio_in, n_fft=NFFT, win_length=NFFT, hop_length=hop_length_FFT, center=True,window=np.blackman(NFFT))

                    Pxx = np.concatenate([np.real(Pxx_c), np.imag(Pxx_c)], axis=0)

                    # plt.imshow((Pxx), aspect='auto', vmin=0, vmax=10)
                    # plt.colorbar()
                    # plt.show()

                    if file_audio_index == 1:
                        spec_width = Pxx.shape[0]
                        spec_length = Pxx.shape[1]
                        tensor_shape = (spec_width, spec_length)
                        Y_test = np.reshape(Pxx, tensor_shape)
                    else:
                        Y_test = np.concatenate((Y_test, np.reshape(Pxx, tensor_shape)),axis=1)

                    S = librosa.feature.melspectrogram(S=Pxx_c, sr=fs, n_mels=128, n_fft=NFFT,
                                                       hop_length=hop_length_FFT)
                    MFCCxx = librosa.feature.mfcc(S=librosa.core.logamplitude(S), n_mfcc=N_MFCC, sr=fs)

                    # MFCCxx = librosa.feature.mfcc(S=Pxx_c, sr=fs, n_mfcc=N_MFCC)

                    if file_audio_index == 1:
                        mfcc_shape = (MFCCxx.shape[0], MFCCxx.shape[1])
                        X_test = np.reshape(MFCCxx, mfcc_shape)
                    else:
                        X_test = np.append(X_test, np.reshape(MFCCxx, mfcc_shape), axis=1)
                    file_audio_index = file_audio_index + 1
                else:
                    continue
                continue

        if TRAIN_TEST == 1:
            pickle.dump(X_train, open("%s/%s" % (training_dataset_dir, 'TRAINING_DATASET.p'), 'wb'))
            pickle.dump(Y_train, open("%s/%s" % (training_dataset_dir, 'TRAINING_LABELS.p'), 'wb'))
            np.savetxt(training_dataset_dir + '/filenames.txt', filename_list_training,fmt='%s', newline='\n')

        elif TRAIN_TEST == 0:
            # X_train = pickle.load(open("%s/%s" % (training_dataset_dir, 'TRAINING_DATASET.p'), 'rb'))
            # X_train = np.reshape(X_train, tensor_shape)
            Y_train = pickle.load(open("%s/%s" % (training_dataset_dir, 'TRAINING_LABELS.p'), 'rb'))
            # Y_train = np.reshape(Y_train, mfcc_shape)
            X_train = np.zeros(1)

        pickle.dump(X_test, open("%s/%s" % (testing_dataset_dir, 'TESTING_DATASET.p'), 'wb'))
        pickle.dump(Y_test, open("%s/%s" % (testing_dataset_dir, 'TESTING_LABELS.p'), 'wb'))
        np.savetxt(testing_dataset_dir + '/filenames.txt', filename_list_testing, fmt='%s', newline='\n')

    else:
        print 'Loading dataset...'
        tensor_shape = pickle.load(open(stft_shape_file_name, 'rb'))
        mfcc_shape = pickle.load(open(mfcc_shape_file_name, 'rb'))

        if TRAIN_TEST == 1:
            X_train = pickle.load(open("%s/%s" % (training_dataset_dir, 'TRAINING_DATASET.p'), 'rb'))
            X_test = pickle.load(open("%s/%s" % (testing_dataset_dir, 'TESTING_DATASET.p'), 'rb'))

            Y_train = pickle.load(open("%s/%s" % (training_dataset_dir, 'TRAINING_LABELS.p'), 'rb'))
            Y_test = pickle.load(open("%s/%s" % (testing_dataset_dir, 'TESTING_LABELS.p'), 'rb'))

        elif TRAIN_TEST == 0:
            X_train = np.zeros(1)
            Y_train = pickle.load(open("%s/%s" % (training_dataset_dir, 'TRAINING_LABELS.p'), 'rb'))

            X_test = pickle.load(open("%s/%s" % (testing_dataset_dir, 'TESTING_DATASET.p'), 'rb'))
            Y_test = pickle.load(open("%s/%s" % (testing_dataset_dir, 'TESTING_LABELS.p'), 'rb'))

    print '\nDataset Extracted:'
    print 'X_train shape', X_train.shape
    print 'Y_train shape', Y_train.shape

    print 'X_test shape', X_test.shape
    print 'Y_test shape', Y_test.shape
    # print midi_note_testing_vector

    return X_train, Y_train, X_test, Y_test, mfcc_shape, tensor_shape


def normalization_feature(feature, feature_name, reshape_size, feature_range_normalization, LABELS, TR, NORM=True):

    from PATH_VAR_INIT import dataset_name,dataset_type, current_directory
    scaler = preprocessing.MinMaxScaler(feature_range=feature_range_normalization, copy=False)

    if TR==False:
        f = open(current_directory+'/DATASET_STATISTICS/DATASET_' + dataset_name+dataset_type + '_STAT_' + feature_name + '.txt', 'rb')
        # mean=np.array([float(f.readline())])
        max = np.array([float(f.readline())])
        data_min = np.array([float(f.readline())])
        range = np.array([float(f.readline())])
        std = np.array([float(f.readline())])
        min = np.array([float(f.readline())])
        f.close()
        # inputScaler_2.mean_ = mean
        scaler.scale_ = std
        scaler.data_min_ = data_min
        scaler.data_range_ = range
        scaler.data_max_ = max
        scaler.min_ = min

    if NORM:
        if TR:
            if LABELS == False:
                feature = np.reshape(feature,[-1,1])
            feature = scaler.fit_transform(feature)
            if LABELS == False:
                feature = np.reshape(feature, [-1,reshape_size])

            std = scaler.scale_[0]
            max = scaler.data_max_[0]
            data_min = scaler.data_min_[0]
            min = scaler.min_[0]
            range = scaler.data_range_[0]
            f = open(current_directory+'/DATASET_STATISTICS/DATASET_' + dataset_name+dataset_type + '_STAT_'+feature_name+'.txt', 'wb')
            # f.write("%s"%mean)
            f.write("%s\n" % max)
            f.write("%s\n" % data_min)
            f.write("%s\n" % range)
            f.write("%s\n" % std)
            f.write("%s\n" % min)
            f.close()
        else:

            if LABELS == False:
                feature = np.reshape(feature, [-1, 1])
            feature = scaler.transform(feature)
            if LABELS == False:
                feature = np.reshape(feature, [-1, reshape_size])
    elif NORM==False:
        feature = scaler.inverse_transform(feature)

    return feature


def normalization_Labels(Y_train, Y_test, dataset_type, current_directory, feature_name='labels', feature_range_normalization=(-1,1), NORM=True):

    labelsScaler = preprocessing.MinMaxScaler(feature_range=feature_range_normalization, copy=False)

    if NORM:

        Y_train = labelsScaler.fit_transform(Y_train)
        Y_test = labelsScaler.transform(Y_test)

    std = labelsScaler.scale_[0]
    max = labelsScaler.data_max_[0]
    data_min = labelsScaler.data_min_[0]
    min = labelsScaler.min_[0]
    range = labelsScaler.data_range_[0]
    f = open(current_directory+'/DATASET_STATISTICS/DATASET_' + dataset_type + '_STAT_' + feature_name + '.txt', 'wb')
    # f.write("%s"%mean)
    f.write("%s\n" % max)
    f.write("%s\n" % data_min)
    f.write("%s\n" % range)
    f.write("%s\n" % std)
    f.write("%s\n" % min)
    f.close()

    return Y_train,Y_test,labelsScaler


def plot(network_test_output,Y_test, diff,diff_norm, output_recap_file,current_directory, tensor_shape, xmax=1000, PC_TRAINING_TESTING=PC_TRAINING_TESTING):

    out_file = open(current_directory + '/' + output_recap_file, "w")
    out_file.write("Mean difference = %s\n" % np.mean(diff))
    out_file.write("Mean difference NORMALIZED = %s\n" % np.mean(diff_norm))
    out_file.write("RMS difference = %s\n" % np.sqrt(np.mean(diff ** 2)))
    out_file.close()

    network_test_output =np.abs(network_test_output[:,0:tensor_shape[0]/2]+1j*network_test_output[:,tensor_shape[0]/2:tensor_shape[0]])
    Y_test = np.abs(Y_test[:,0:tensor_shape[0]/2]+1j*Y_test[:,tensor_shape[0]/2:tensor_shape[0]])

    plt.suptitle('STFT cfr')
    plt.subplot(211)
    plt.title('Estimated STFT from MFCC')
    plt.imshow(network_test_output[0:xmax, :], aspect='auto',vmin=-1,vmax=10)
    plt.colorbar()
    plt.subplot(212)
    plt.title('Original STFT')
    plt.imshow(Y_test[0:xmax, :], aspect='auto',vmin=-1,vmax=10)
    plt.colorbar()
    if PC_TRAINING_TESTING:
        plt.show()

    plt.savefig(current_directory+'/Error on TEST set.png')
    plt.clf()


def netOut2Spec(network_test_output, tensor_shape):

    spec_mat = network_test_output[:,0:tensor_shape[0]/2]+1j*network_test_output[:,tensor_shape[0]/2:tensor_shape[0]]
    spec_mat = np.reshape(spec_mat, [-1, tensor_shape[1], tensor_shape[0] / 2])

    return spec_mat


def spec2Audio(audio_mat,audio_mat_target, spec_mat, file_name, folder, fs, NFFT, hop_length, spec_target, PLOT=True, SAVE=True):

    fig_ratio = (10,5) # width, height

    f = open(file_name,'rb')
    filenames=f.readlines()
    f.close()

    if os.path.exists(folder):
        os.system('rm -r '+folder)
        os.makedirs(folder)
        os.system('chmod -R 777 '+folder)

    if not os.path.exists(folder):
        os.makedirs(folder)
        os.system('chmod -R 777 '+folder)

    for i in xrange(spec_mat.shape[0]):

        audio = audio_mat[i,:]
        scaled = np.int16(audio / np.max(np.abs(audio)) * 32767)
        fpath = folder + '/' + filenames[i]
        if fpath.endswith('\n'):
            fpath=fpath[0:len(fpath)-1]
        if SAVE:
            WAV.write(fpath, fs, scaled)

        if PLOT:

            plt.figure(figsize=fig_ratio, tight_layout=True)
            plt.subplot(221)
            plt.title('Target STFT')
            plt.imshow(np.abs(spec_target[i,:,:].T),aspect='auto', cmap='gray_r')
            plt.xlabel("Time [Frames]", fontsize=14)
            plt.ylabel("Frequency [Hz]", fontsize=14)
            plt.tick_params(axis='both', which='major', labelsize=14)

            plt.subplot(222)
            plt.title('Estimated STFT')
            plt.imshow(np.abs(spec_mat[i, :, :].T),aspect='auto', cmap='gray_r')
            plt.xlabel("Time [Frames]", fontsize=14)
            plt.ylabel("Frequency [Hz]", fontsize=14)
            plt.tick_params(axis='both', which='major', labelsize=14)

            plt.subplot(223)
            plt.title('Target waveform')
            audio_t = audio_mat_target[i,:]
            plt.plot(paper_spectr_plot.normalize(audio_t),'k')
            plt.ylim([-1,1])
            plt.xlabel("Time [Samples]", fontsize=14)
            plt.ylabel("Amplitude", fontsize=14)
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.grid(True)

            plt.subplot(224)
            plt.title('Estimated waveform')
            plt.plot(paper_spectr_plot.normalize(audio),'k')
            plt.ylim([-1, 1])
            plt.xlabel("Time [Samples]", fontsize=14)
            plt.ylabel("Amplitude", fontsize=14)
            plt.tick_params(axis='both', which='major', labelsize=10)
            plt.grid(True)

            plt.suptitle('Estimated STFT vs Target STFT '+filenames[i][0:8], fontsize=16)
            figname = fpath[0:len(fpath)-4]+'.png'
            plt.savefig(figname)
            # plt.show()
            plt.clf()
            plt.close()


def main():
    import random
    random.seed(6)
    np.random.seed(6)

    from PATH_VAR_INIT import current_directory, NFFT, hop_length_FFT, fs, N_MFCC, training_dataset_dir, testing_dataset_dir, audio_format, audio_reshape, audio_reshape_size, MAKE_DATASET, TRAIN_TEST
    from PATH_VAR_INIT import activations, batchnormalization, learn_rate,decay,MomentumMax,optimizer
    from PATH_VAR_INIT import trained_network, ES_epochs, minibatch_size, maxEpochs,VALIDATION, LOAD_MODEL
    from PATH_VAR_INIT import dataset_type,dataset_name,NORMALIZATION, DENORMALIZATION
    from PATH_VAR_INIT import systematic_save_output_folder, output_recap_file, history_file, NETWORK_TYPE
    from PATH_VAR_INIT import context, final_layer_activation, CUSTOM_COST_FUNCTION, feature_range_normalization

    if NETWORK_TYPE == 'MLP':
        print 'Detected MLP as NETWORK_TYPE'
        from PATH_VAR_INIT import N_Layers, layer_dim, dropout_vec

        mfcc_shape_file_name=current_directory+'/mfcc_shape_file.p'
        stft_shape_file_name= current_directory+'/stft_shape_file.p'
        X_train, Y_train, X_test, Y_test, mfcc_shape, tensor_shape = dataExtraction(NFFT, hop_length_FFT, fs, N_MFCC, training_dataset_dir, testing_dataset_dir, audio_format,
                       mfcc_shape_file_name, stft_shape_file_name, audio_reshape, audio_reshape_size,
                       MAKE_DATASET, TRAIN_TEST)

        X_train = X_train.T
        X_test = X_test.T
        Y_train = Y_train.T
        Y_test = Y_test.T

        if NORMALIZATION == 1:

            if TRAIN_TEST:
                X_train = normalization_feature(X_train, 'X_train', N_MFCC, (feature_range_normalization), LABELS=False, TR=True)
                if context > 0:
                    X_train = NetCreate.ctx(np.reshape(X_train,[-1,mfcc_shape[0], mfcc_shape[1]]), ctx_fr=context, RESHAPE=True)[0]

            X_test= normalization_feature(X_test, 'X_train', N_MFCC, (feature_range_normalization), LABELS=False, TR=False)

            Y_train, Y_test, labelsScaler = normalization_Labels(Y_train, Y_test, dataset_name+dataset_type, current_directory,feature_name='labels', feature_range_normalization=(-1, 1), NORM=True)

        if context>0:
            # plt.subplot(121)
            # plt.imshow(X_test,aspect='auto')
            X_test=NetCreate.ctx(np.reshape(X_test,[-1,mfcc_shape[0], mfcc_shape[1]]),ctx_fr=context,RESHAPE=True)[0]
            # plt.subplot(122)
            # plt.imshow(X_test, aspect='auto')
            # plt.show()

        print '\nInput sizes with context:'
        print '\nX_train: ', X_train.shape
        print 'X_test: ', X_test.shape

        model = NetCreate.mainMLP_custom_cost(N_Layers, int(X_test.shape[1]), layer_dim, activations, final_layer_activation, batchnormalization, dropout_vec, learn_rate,decay,MomentumMax,optimizer,CUSTOM_COST_FUNCTION=CUSTOM_COST_FUNCTION,custom_cost_function_name=custom_cost_function)

        network_test_output, history = NetCreate.training_testing(X_train, Y_train, X_test, model, TRAIN_TEST, trained_network, ES_epochs, minibatch_size, maxEpochs,VALIDATION, LOAD_MODEL=LOAD_MODEL,CUSTOM_COST_FUNCTION=CUSTOM_COST_FUNCTION,custom_cost_function_name=custom_cost_function)

        if TRAIN_TEST == 1:
            PostUtil.history_plot(history, current_directory, PLT_SHOW=PC_TRAINING_TESTING)

        diff_norm = np.abs(network_test_output - Y_test)

        if NORMALIZATION == 1 and DENORMALIZATION == 1:
            Y_test = labelsScaler.inverse_transform(Y_test)
            network_test_output = labelsScaler.inverse_transform(network_test_output)

        diff=np.abs(network_test_output-Y_test)

        # np.savetxt(current_directory + '/' + 'network_test_output.txt', network_test_output)
        # np.savetxt(current_directory + '/' + 'network_target_output.txt', Y_test)

        plot(network_test_output, Y_test, diff, diff_norm, output_recap_file, current_directory, tensor_shape=tensor_shape, xmax=4000)

        spec_mat = netOut2Spec(network_test_output,tensor_shape)
        spec_target = netOut2Spec(Y_test,tensor_shape)

        spec2Audio(spec_mat, spec_target=spec_target, fs=fs, NFFT=NFFT, hop_length=hop_length_FFT, file_name=testing_dataset_dir+'/filenames.txt',folder=current_directory+'/Networks/Audio_Output', SAVE=True)

        PostUtil.systematic_test(output_folder=systematic_save_output_folder,trained_network=trained_network, output_recap_file=output_recap_file, history_file=history_file,  network_test_output='network_test_output.txt', network_target_output='network_target_output.txt',TRAIN_TEST=TRAIN_TEST,NETWORK_TYPE=NETWORK_TYPE, testing_register_type='Audio_Output',ERROR_AVE=np.mean(diff_norm),current_directory=current_directory,PATH_VAR_INIT_ref=PATH_VAR_INIT_ref)

    elif NETWORK_TYPE == 'AE':
        print 'Detected AE as NETWORK_TYPE'
        from PATH_VAR_INIT import N_Layers, frame_length, layer_dim, N_layers_MLP,N_Layers_Conv,maxPooling,N_filters,CNN_kernel_sizes

        X_train, Y_train, X_test, Y_test = dataExtraction_MFCC_Recon(NFFT, hop_length_FFT, fs, N_MFCC, training_dataset_dir, testing_dataset_dir,
                                      audio_format, audio_reshape, audio_reshape_size, MAKE_DATASET, TRAIN_TEST)

        X_train = paper_spectr_plot.normalize(X_train)
        X_test = paper_spectr_plot.normalize(X_test)
        Y_train = paper_spectr_plot.normalize(Y_train)
        Y_test = paper_spectr_plot.normalize(Y_test)

        # if NORMALIZATION == 1:
        #
        #     if TRAIN_TEST:
        #         X_train = normalization_feature(X_train, 'X_train', X_train.shape[1], (feature_range_normalization),
        #                                         LABELS=False, TR=True)
        #         Y_train = normalization_feature(Y_train, 'Y_train', X_train.shape[1], (feature_range_normalization),
        #                                     LABELS=False, TR=True)
        #
        #     X_test= normalization_feature(X_test, 'X_train', X_test.shape[1], (feature_range_normalization), LABELS=False, TR=False)
        #     Y_test= normalization_feature(Y_test, 'Y_train', X_test.shape[1], (feature_range_normalization), LABELS=False, TR=False)
        #
        #     # Y_train, Y_test, labelsScaler = normalization_Labels(Y_train, Y_test, dataset_name+dataset_type, current_directory,feature_name='labels', feature_range_normalization=(feature_range_normalization), NORM=True)

        # autoencoder = NetCreate.MainAE(N_Layers, frame_length, layer_dim, activations, batchnormalization, learn_rate, decay, MomentumMax, optimizer)

        autoencoder = NetCreate.MainAE_CNN(N_Layers_Conv, N_layers_MLP, frame_length, layer_dim, maxPooling, N_filters, CNN_kernel_sizes,
               activations, batchnormalization, learn_rate, decay, MomentumMax, optimizer)
        autoencoder.summary()

        if TRAIN_TEST==1:
            X_train = np.reshape(X_train,[-1,frame_length,1])

        network_test_output, history = NetCreate.training_testing(X_train, np.reshape(Y_train,[-1,frame_length]), np.reshape(X_test,[-1,frame_length,1]), autoencoder, TRAIN_TEST,trained_network, ES_epochs, minibatch_size, maxEpochs, VALIDATION, LOAD_MODEL=LOAD_MODEL,CUSTOM_COST_FUNCTION=False,custom_cost_function_name=custom_cost_function)

        if TRAIN_TEST == 1:
            PostUtil.history_plot(history, current_directory, PLT_SHOW=PC_TRAINING_TESTING)

        network_test_output = np.reshape(network_test_output,Y_test.shape)
        diff_norm = np.abs(network_test_output - Y_test)

        # if NORMALIZATION == 1 and DENORMALIZATION == 1:
        #     Y_test = labelsScaler.inverse_transform(Y_test)
        #     network_test_output = labelsScaler.inverse_transform(network_test_output)

        for i in xrange(len(network_test_output)-1):
            if i == 0:
                spec_mat = librosa.core.stft(network_test_output[i],n_fft=NFFT, win_length=NFFT, hop_length=hop_length_FFT, center=True,window=np.blackman(NFFT))
                spec_target = librosa.core.stft(Y_test[i],n_fft=NFFT, win_length=NFFT, hop_length=hop_length_FFT, center=True,window=np.blackman(NFFT))
                spec_mat = np.reshape(spec_mat,[1,spec_mat.shape[0],spec_mat.shape[1]])
                spec_target = np.reshape(spec_target,[1,spec_target.shape[0],spec_target.shape[1]])

            spec_mat = np.row_stack((spec_mat, np.reshape(librosa.core.stft(network_test_output[i],n_fft=NFFT, win_length=NFFT, hop_length=hop_length_FFT, center=True,window=np.blackman(NFFT)),[1,spec_mat.shape[1],spec_mat.shape[2]])))
            spec_target = np.row_stack((spec_target, np.reshape(librosa.core.stft( Y_test[i],n_fft=NFFT, win_length=NFFT, hop_length=hop_length_FFT, center=True,window=np.blackman(NFFT)),[1,spec_target.shape[1],spec_target.shape[2]])))
        spec_mat=np.reshape(spec_mat,[spec_mat.shape[0],spec_mat.shape[2],spec_mat.shape[1]])
        spec_target=np.reshape(spec_target,[spec_target.shape[0],spec_target.shape[2],spec_target.shape[1]])

        spec2Audio(audio_mat=network_test_output, audio_mat_target=Y_test, spec_mat=spec_mat, spec_target=spec_target, fs=fs, NFFT=NFFT, hop_length=hop_length_FFT,file_name=testing_dataset_dir + '/filenames.txt',folder=current_directory + '/Networks/Audio_Output', SAVE=True)

        PostUtil.systematic_test(output_folder=systematic_save_output_folder, trained_network=trained_network,
                                 output_recap_file=output_recap_file, history_file=history_file, network_test_output='None',
                                 network_target_output='None', TRAIN_TEST=TRAIN_TEST,
                                 NETWORK_TYPE=NETWORK_TYPE, testing_register_type='Audio_Output',
                                 ERROR_AVE=np.mean(diff_norm), current_directory=current_directory,
                                 PATH_VAR_INIT_ref=PATH_VAR_INIT_ref)


if __name__ == '__main__':
    main()
